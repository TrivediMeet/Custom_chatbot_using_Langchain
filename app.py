from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from bs4 import BeautifulSoup
import requests
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

app = Flask(__name__)
api = Api(app)

URL = "https://brainlox.com/courses/category/technical"

# Global variable to store course data and vector store
course_data = []
vector_store = None

def extract_course_data():
    global course_data
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    response = requests.get(URL, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Debug print to see the raw HTML
    print("Raw HTML:", soup.prettify())

    courses = []

    # Extract course information based on the provided HTML structure
    for course in soup.find_all('div', class_='single-courses-box'):
        title = course.find('h3').text.strip() if course.find('h3') else 'No title'
        description = course.find('p').text.strip() if course.find('p') else 'No description'
        
        courses.append({
            'title': title,
            'description': description
        })

    if not courses:
        print("No courses found. Verify the HTML structure.")
    else:
        print("Extracted Courses:", courses)
    
    course_data = courses
// For creae embeddings
def create_embeddings():
    global vector_store
    if not course_data:
        print("No course data available.")
        return
    
    embeddings = OpenAIEmbeddings()  # Make sure you have the OPENAI_API_KEY set in your environment
    vectors = embeddings.embed_documents([course['description'] for course in course_data])  # Create embeddings for all descriptions
    
    # Convert to numpy array
    vectors = np.array(vectors).astype(np.float32)

    # Create a FAISS index
    index = faiss.IndexFlatL2(len(vectors[0]))  # L2 distance index
    index.add(vectors)  # Add vectors to the index
    vector_store = FAISS(index, embeddings)

    print("Embeddings created and stored in vector store.")

class EmbeddingResource(Resource):
    def post(self):
        extract_course_data()  # Extract course data
        create_embeddings()     # Create embeddings
        return jsonify({"message": "Embeddings created successfully."})

class QueryResource(Resource):
    def post(self):
        user_query = request.json.get('query')
        if not user_query:
            return jsonify({"error": "Query not provided."}), 400
        
        if not vector_store:
            return jsonify({"error": "Embeddings not created. Please call /embeddings first."}), 400
        
        # Retrieve similar courses based on the user's query
        query_vector = vector_store.embeddings.embed_documents([user_query])[0]
        distances, indices = vector_store.similarity_search(query_vector, k=5)  # Get top 5 results
        
        results = []
        for idx in indices:
            results.append(course_data[idx])
        
        return jsonify(results)

api.add_resource(EmbeddingResource, '/embeddings')
api.add_resource(QueryResource, '/query')

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
