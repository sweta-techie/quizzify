import streamlit as st
import PyPDF2
from transformers import pipeline
import numpy as np
from sentence_transformers import SentenceTransformer, util

class DocumentProcessor:
    def __init__(self):
        self.processed_data = []

    def ingest_documents(self, documents):
        for document in documents:
            self.process_pdf(document)

    def process_pdf(self, document):
        try:
            reader = PyPDF2.PdfReader(document)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            self.processed_data.append({"filename": document.name, "text": text})
            st.write(f"Extracted text from {document.name}:\n{text[:500]}...")
        except Exception as e:
            st.error(f"Failed to process {document.name}: {e}")

class EmbeddingClient:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, text):
        return self.model.encode(text)

class ChromaCollectionCreator:
    def __init__(self, document_processor, embedding_client):
        self.document_processor = document_processor
        self.embedding_client = embedding_client
        self.collection = []
        self.question_generator = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")

    def create_collection(self, topic):
        st.write(f"Creating Chroma collection for topic '{topic}'...")
        for doc in self.document_processor.processed_data:
            embedding = self.embedding_client.get_embeddings(doc['text'])
            self.collection.append({"filename": doc['filename'], "embedding": embedding, "text": doc['text']})
        st.write(f"Chroma collection created with {len(self.collection)} documents.")

    def query_collection(self, query):
        st.write(f"Querying Chroma collection with: {query}")
        query_embedding = self.embedding_client.get_embeddings(query)
        results = []

        # Perform semantic search and relevance scoring
        for item in self.collection:
            similarity = util.pytorch_cos_sim(query_embedding, item['embedding'])[0]
            st.write(f"Similarity with document {item['filename']}: {similarity.item()}")
            if similarity.item() > 0.3:  # Adjust the threshold as needed
                results.append((similarity.item(), item))

        # Sort results based on similarity score
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return [item[1] for item in results]

    def generate_questions(self, chunks, num_questions):
        st.write("Generating questions based on the selected chunks...")
        questions = []

        for chunk in chunks:
            split_chunks = self.split_text_into_chunks(chunk)
            for split_chunk in split_chunks:
                if len(split_chunk.strip()) > 0:
                    st.write(f"Attempting to generate question from chunk: {split_chunk.strip()[:100]}...")  # Display the first 100 characters
                    try:
                        generated_questions = self.question_generator(split_chunk, max_length=50, do_sample=True)
                        for question in generated_questions:
                            st.write(f"Generated Question: {question['generated_text']}")
                            questions.append(question['generated_text'])
                            if len(questions) >= num_questions:
                                return questions
                    except Exception as e:
                        st.error(f"Error generating question: {e}")
        return questions

    def split_text_into_chunks(self, text, max_length=512):
        # Split text into chunks of a specified maximum length
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + '. '
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

# Streamlit UI setup (remains the same as before)
st.title("PDF Document Processor with Query and Question Generation")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        st.session_state.processor = DocumentProcessor()
        st.session_state.processor.ingest_documents(uploaded_files)
    else:
        st.error("Please upload at least one PDF file.")

embed_config = {
    "model_name": "all-MiniLM-L6-v2"
}

if st.button("Initialize Embedding Client"):
    st.session_state.embedding_client = EmbeddingClient(
        model_name=embed_config["model_name"]
    )
    st.write("Embedding Client initialized.")

if st.button("Create Chroma Collection"):
    if st.session_state.processor and st.session_state.embedding_client:
        st.session_state.chroma_creator = ChromaCollectionCreator(
            document_processor=st.session_state.processor,
            embedding_client=st.session_state.embedding_client
        )
        st.session_state.chroma_creator.create_collection("Sample Topic")
    else:
        st.error("Please process documents and initialize the EmbeddingClient first.")

# Query input and question generation
query = st.text_input("Enter your query related to the quiz topic:")

if st.button("Search and Generate Questions"):
    if st.session_state.chroma_creator:
        relevant_chunks = st.session_state.chroma_creator.query_collection(query)
        if relevant_chunks:
            chunk_texts = [item['text'] for item in relevant_chunks]
            num_questions = st.slider("Select the number of questions to generate:", min_value=1, max_value=20, value=5)
            questions = st.session_state.chroma_creator.generate_questions(chunk_texts, num_questions)
            if questions:
                st.write("Generated Quiz Questions:")
                for i, question in enumerate(questions):
                    st.write(f"{i+1}. {question}")
            else:
                st.write("No questions could be generated.")
        else:
            st.write("No relevant information found for the query.")
    else:
        st.error("Please create the Chroma collection first.")
