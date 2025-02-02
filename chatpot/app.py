import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document  # Add this import
from PIL import Image
import ollama
import time
import pytesseract
from pathlib import Path
from typing import List

class OllamaEmbeddings:
    def __init__(self, model_name="deepseek-r1"):
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        embeddings = []
        for text in texts:
            embedding = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(embedding['embedding'])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text."""
        embedding = ollama.embeddings(model=self.model_name, prompt=text)
        return embedding['embedding']
    
    def __call__(self, text: str) -> List[float]:
        """Make the class callable."""
        return self.embed_query(text)

class ImageLoader:
    def __init__(self, directory):
        self.directory = directory
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    
    def load(self):
        documents = []
        for file_path in Path(self.directory).glob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    image = Image.open(file_path)
                    text = pytesseract.image_to_string(image)
                    
                    if text.strip():
                        # Create a proper Document object
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': str(file_path),
                                'type': 'image'
                            }
                        )
                        documents.append(doc)
                except Exception as e:
                    st.warning(f"Error processing image {file_path}: {str(e)}")
                    
        return documents

def get_response_from_ollama(prompt):
    """Get response from Ollama API with proper response handling."""
    try:
        response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
        # Access the 'message' dictionary and then get the 'content' field
        return response.message['content']
    except Exception as e:
        st.error(f"Error getting response from Ollama: {str(e)}")
        # Return a default message in case of error
        return "I apologize, but I encountered an error processing your request."

def vector_embedding():
    if "vectors" not in st.session_state:
        documents = []
        
        # Load PDF documents
        if os.path.exists("./documents"):
            loader = PyPDFDirectoryLoader("./documents")
            # PyPDFDirectoryLoader already returns proper Document objects
            pdf_docs = loader.load()
            st.info(f"Found {len(pdf_docs)} PDF documents")
            documents.extend(pdf_docs)
        
        # Load images
        if os.path.exists("./images"):
            image_loader = ImageLoader("./images")
            image_docs = image_loader.load()
            st.info(f"Found {len(image_docs)} images with text content")
            documents.extend(image_docs)
            
        if not documents:
            st.error("No documents or images found. Please add files to 'documents' or 'images' directory.")
            return

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)

        if not final_documents:
            st.error("No content could be extracted from the files.")
            return

        st.info(f"Created {len(final_documents)} content chunks")

        try:
            # Create embedding object and vector store
            embedding_model = OllamaEmbeddings()
            texts = [doc.page_content for doc in final_documents]
            metadatas = [doc.metadata for doc in final_documents]  # Use full metadata
            
            # Create vector store using the embedding model instance
            st.session_state.vectors = FAISS.from_texts(
                texts=texts,
                embedding=embedding_model,
                metadatas=metadatas
            )
            st.success("Vector Store DB Is Ready")
            st.session_state.final_documents = final_documents
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            raise  # Re-raise the exception to see the full traceback
    else:
        st.info("Vector store is already initialized.")

def main():
    st.title("Model Document & Image Q&A")

    prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """)

    # Add a reset button
    if st.button("Reset Vector Store"):
        if "vectors" in st.session_state:
            del st.session_state.vectors
        if "final_documents" in st.session_state:
            del st.session_state.final_documents
        st.success("Vector store reset successfully")

    # User input for question
    prompt1 = st.text_input("Enter Your Question About Documents and Images")

    # Create vector store and embed documents
    if st.button("Process Documents and Images"):
        vector_embedding()

    # Handle question answering
    if prompt1:
        if "vectors" in st.session_state:
            try:
                retriever = st.session_state.vectors.as_retriever()

                def document_chain_fn(inputs):
                    context = inputs["context"]
                    question = inputs["input"]
                    prompt_with_context = prompt.format(context=context, input=question)
                    answer = get_response_from_ollama(prompt_with_context)
                    return {"answer": answer}

                retrieval_chain = create_retrieval_chain(retriever, document_chain_fn)

                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})

                st.write(f"Response time: {time.process_time() - start} seconds")
                st.write(response['answer'])

                # Display document similarity search results in expander
                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response.get("context", [])):
                        st.write(f"Document {i + 1}:")
                        st.write(f"Source: {doc['metadata']['source']}")
                        st.write(doc['page_content'])
                        st.write("--------------------------------")
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                raise  # Re-raise the exception to see the full traceback
        else:
            st.warning("Vector store is not initialized. Please click 'Process Documents and Images' first.")

if __name__ == "__main__":
    main()