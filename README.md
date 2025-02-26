# SENSEI-SEARCH: IBM Document & Image Q&A

SENSEI-SEARCH is a Streamlit application that allows users to ask questions about documents and images processed using IBM Watsonx.ai. It provides intelligent answers based on the content, leveraging IBM Cloud services for text generation.

## Features

- **IBM Watsonx.ai Integration**: Uses IBM Watson for document-based Q&A
- **Text Generation**: Provides accurate, AI-driven responses to user queries
- **Seamless Interaction**: Easy-to-use UI for interacting with the system

## Prerequisites

- Python 3.7+
- IBM Cloud API Key and Deployment ID (for IBM Watsonx.ai)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/BhushanSah3/Sensei_IBM
   cd sensei-search
   ```

2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Obtain IBM Cloud Credentials:**
   - Get your **IBM API Key** and **Deployment ID** from your IBM Cloud account (Watsonx.ai service).

## Usage

1. **Run the Application:**
   ```bash
   cd chatpot
   streamlit run ibm.py
   ```

2. **Enter Your Credentials:**
   - Input your **IBM API Key** and **Deployment ID** in the provided fields on the UI.

3. **Ask Questions:**
   - Once credentials are verified, you can ask questions related to documents and images.

4. **View the Response:**
   - The application will return a context-aware answer generated by IBM Watsonx.ai.
