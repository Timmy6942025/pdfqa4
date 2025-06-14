# main.py
import os
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import GooglePalm
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
from langchain_core.language_models.llms import LLM
from typing import Any, List
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Gemini LLM wrapper for LangChain
class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash-preview-05-20"
    api_key: str = os.getenv("GOOGLE_API_KEY")

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

llm = GeminiLLM()
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global chat_history
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Get relevant documents from vector store
        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Create context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Build chat history context
        history_context = ""
        if chat_history:
            history_context = "\n\nPrevious conversation:\n"
            for q, a in chat_history[-3:]:  # Last 3 exchanges
                history_context += f"Q: {q}\nA: {a}\n"
        
        # Create prompt for Gemini
        prompt = f"""Based on the following context from the document, please answer the question.

Context:
{context}

{history_context}

Question: {question}

Please provide a helpful and accurate answer based on the provided context. If the answer cannot be found in the context, please say so."""

        # Get response from Gemini
        answer = llm(prompt)
        
        chat_history.append((question, answer))
        chat_history = chat_history[-5:]  # Keep last 5 exchanges
        
        return jsonify({'answer': answer})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
