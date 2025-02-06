import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from PIL import Image
import ollama
import time
import pytesseract
import speech_recognition as sr
from pydub import AudioSegment
import io
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    def __init__(self, model_name="deepseek-r1"):
        self.model_name = model_name
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            embedding = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(embedding['embedding'])
        return embeddings
    
    def embed_query(self, text):
        embedding = ollama.embeddings(model=self.model_name, prompt=text)
        return embedding['embedding']
    
    def __call__(self, text):
        return self.embed_query(text)

def process_image(image_file):
    """Process uploaded image file and extract text."""
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        
        if text.strip():
            return Document(
                page_content=text,
                metadata={
                    'source': image_file.name,
                    'type': 'image'
                }
            )
        return None
    except Exception as e:
        logger.error(f"Error processing image {image_file.name}: {str(e)}")
        st.error(f"Error processing image {image_file.name}: {str(e)}")
        return None

def process_audio(audio_file):
    """Process uploaded audio file and transcribe to text."""
    try:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name

        # Convert audio to WAV format if needed
        audio = AudioSegment.from_file(tmp_path)
        audio.export(tmp_path, format="wav")

        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Read the audio file
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            
        # Transcribe audio to text
        text = recognizer.recognize_google(audio_data)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        if text.strip():
            return Document(
                page_content=text,
                metadata={
                    'source': audio_file.name,
                    'type': 'audio'
                }
            )
        return None
    except Exception as e:
        logger.error(f"Error processing audio {audio_file.name}: {str(e)}")
        st.error(f"Error processing audio {audio_file.name}: {str(e)}")
        return None

def get_response_from_ollama(prompt):
    """Get direct response from Ollama API."""
    try:
        response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
        return response.message['content']
    except Exception as e:
        logger.error(f"Error getting response from Ollama: {str(e)}")
        st.error(f"Error getting response from Ollama: {str(e)}")
        return "I apologize, but I encountered an error processing your request."

def initialize_or_get_vector_store():
    """Initialize vector store if there are uploaded files, otherwise return None."""
    documents = []
    
    # Get uploaded files from session state if they exist
    uploaded_images = st.session_state.get('uploaded_images', [])
    uploaded_audio = st.session_state.get('uploaded_audio', [])
    
    # Process images if any
    for image_file in uploaded_images:
        doc = process_image(image_file)
        if doc:
            documents.append(doc)
    
    # Process audio if any
    for audio_file in uploaded_audio:
        doc = process_audio(audio_file)
        if doc:
            documents.append(doc)
    
    if not documents:
        return None

    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)
        
        # Create embedding object and vector store
        embedding_model = OllamaEmbeddings()
        texts = [doc.page_content for doc in final_documents]
        metadatas = [doc.metadata for doc in final_documents]
        
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=metadatas
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        st.error(f"Error creating vector store: {str(e)}")
        return None

def main():
    st.title("Model Document & Image Q&A")
    st.write("Upload files (optional) and ask questions. The system works with or without uploaded files!")

    # Initialize session state for uploaded files
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'uploaded_audio' not in st.session_state:
        st.session_state.uploaded_audio = []

    # File upload section
    with st.expander("Upload Files (Optional)", expanded=True):
        # Image upload
        uploaded_images = st.file_uploader(
            "Upload Images (Optional)",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True,
            key='image_uploader'
        )
        
        # Audio upload
        uploaded_audio = st.file_uploader(
            "Upload Audio Files (Optional)",
            type=['wav', 'mp3', 'm4a'],
            accept_multiple_files=True,
            key='audio_uploader'
        )

        # Update session state with uploaded files
        st.session_state.uploaded_images = uploaded_images if uploaded_images else []
        st.session_state.uploaded_audio = uploaded_audio if uploaded_audio else []

        total_files = len(st.session_state.uploaded_images) + len(st.session_state.uploaded_audio)
        if total_files > 0:
            st.info(f"Total files uploaded: {total_files}")

    # Question input
    user_question = st.text_input("Ask a question (works with or without uploaded files)")

    if user_question:
        # Check if there are any uploaded files
        vector_store = initialize_or_get_vector_store()

        if vector_store:
            # RAG-based response
            try:
                retriever = vector_store.as_retriever()
                prompt = ChatPromptTemplate.from_template("""
                Answer the question based on the provided context and your knowledge.
                <context>
                {context}
                </context>
                Question: {input}
                """)

                def document_chain_fn(inputs):
                    context = inputs["context"]
                    question = inputs["input"]
                    prompt_with_context = prompt.format(context=context, input=question)
                    answer = get_response_from_ollama(prompt_with_context)
                    return {"answer": answer}

                retrieval_chain = create_retrieval_chain(retriever, document_chain_fn)
                
                with st.spinner('Processing your question...'):
                    start = time.process_time()
                    response = retrieval_chain.invoke({'input': user_question})
                    st.write(f"Response time: {time.process_time() - start:.2f} seconds")
                    st.write(response['answer'])

                    # Show sources in expander
                    with st.expander("View Sources"):
                        for i, doc in enumerate(response.get("context", [])):
                            st.write(f"Source {i + 1}:")
                            st.write(f"Type: {doc['metadata']['type']}")
                            st.write(f"File: {doc['metadata']['source']}")
                            st.write("Content Preview:", doc['page_content'][:200] + "...")
                            st.write("---")

            except Exception as e:
                logger.error(f"Error in RAG processing: {str(e)}")
                st.error("Error processing with uploaded files. Falling back to direct response.")
                direct_response = get_response_from_ollama(user_question)
                st.write(direct_response)
        else:
            # Direct LLM response when no files are uploaded
            with st.spinner('Getting response...'):
                direct_response = get_response_from_ollama(user_question)
                st.write(direct_response)

if __name__ == "__main__":
    main()