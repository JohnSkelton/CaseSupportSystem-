from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import sqlite3
import sys

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

class SQLiteLoader:
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name

    def load(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM [case]")
        rows = cur.fetchall()
        conn.close()

        # Format the rows in a way that language model can understand
        documents = [self.format_row(row) for row in rows]

        return documents

    def format_row(self, row):
        # Create a document object with a page_content and metadata attribute
        document = type('', (), {})()
        document.page_content = ' '.join(map(str, row))
        document.metadata = {}  # or whatever metadata you want to assign
        return document

app = Flask(__name__)

# Use an absolute path to the SQLite file
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'cases.db')

app.config['SECRET_KEY'] = "your_secret_key"
db = SQLAlchemy(app)

class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_number = db.Column(db.String(100))
    description = db.Column(db.String(5000))
    solution = db.Column(db.String(5000))
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

# This will create the tables if they do not exist
with app.app_context():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        case_number = request.form.get('case_number')
        description = request.form.get('description')
        solution = request.form.get('solution')
        case = Case(case_number=case_number, description=description, solution=solution)
        db.session.add(case)
        db.session.commit()
        flash("Case added successfully!", "success")
        return redirect(url_for('index'))
    cases = Case.query.order_by(Case.date_created).all()
    return render_template('index.html', cases=cases)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        message = request.form.get('message')
        if PERSIST and os.path.exists("persist"):
            vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
            index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        else:
            loader = SQLiteLoader('cases.db', 'case')
            if PERSIST:
                index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
            else:
                index = VectorstoreIndexCreator().from_loaders([loader])
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        )

        chat_history = []
        result = chain({"question": message, "chat_history": chat_history})
        response = result['answer']

        return render_template('chat.html', response=response)
    return render_template('chat.html')

@app.route('/delete/<int:id>')
def delete(id):
    case = Case.query.get(id)
    if case:
        db.session.delete(case)
        db.session.commit()
        flash("Case deleted successfully!", "success")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
