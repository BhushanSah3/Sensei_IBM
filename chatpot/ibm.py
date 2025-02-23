import streamlit as st
import requests
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for credentials
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'deployment_id' not in st.session_state:
    st.session_state.deployment_id = None
if 'credentials_submitted' not in st.session_state:
    st.session_state.credentials_submitted = False

def get_iam_token(api_key):
    """Retrieve IAM token from IBM Cloud."""
    try:
        url = "https://iam.cloud.ibm.com/identity/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key
        }
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        logger.error(f"Error getting IAM token: {str(e)}")
        return None

def verify_ibm_credentials(api_key, deployment_id):
    """Verify IBM credentials by checking token retrieval."""
    token = get_iam_token(api_key)
    if token:
        return True, "Credentials verified successfully!"
    return False, "Failed to verify credentials. Check API Key."

def get_response_from_ibm(question, api_key, deployment_id):
    """Send the user question to IBM Watsonx.ai and retrieve the response."""
    try:
        # Get IAM token
        iam_token = get_iam_token(api_key)
        if not iam_token:
            return "Error: Unable to retrieve IAM token."

        # Correct API Endpoint for text generation
        url = f"https://us-south.ml.cloud.ibm.com/ml/v1/deployments/{deployment_id}/text/generation?version=2021-05-01"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {iam_token}",
            "Accept": "application/json"
        }

        # **IMPROVED PAYLOAD**
        payload = {
            "input": f"Q: {question}\nA:",
            "parameters": {
                "decoding_method": "sample",  # Alternative: "greedy"
                "max_new_tokens": 100,  # Adjust based on response length needed
                "temperature": 0.7,  # Control randomness (higher = more creative)
                "top_k": 50,  # Sampling from top 50 tokens
                "top_p": 0.9  # Nucleus sampling
            }
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        data = response.json()

        # Extract generated text from response
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0].get("generated_text", "No meaningful response.")

        return "No valid response received from the model."
    
    except Exception as e:
        logger.error(f"Error calling IBM Watson API: {str(e)}")
        return f"Error: {str(e)}"

def main():
    st.title("IBM Document & Image Q&A")
    st.write("Upload files (optional) and ask questions. The system works with or without uploaded files!")

    # Credentials section
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input("Enter IBM API Key", type="password", key="api_key_input")
        with col2:
            deployment_id = st.text_input("Enter IBM Deployment ID", key="deployment_id_input")

        if st.button("Submit Credentials"):
            if not api_key or not deployment_id:
                st.error("Please enter both API Key and Deployment ID")
            else:
                with st.spinner("Verifying credentials..."):
                    success, message = verify_ibm_credentials(api_key, deployment_id)
                    if success:
                        st.session_state.api_key = api_key
                        st.session_state.deployment_id = deployment_id
                        st.session_state.credentials_submitted = True
                        st.success(message)
                    else:
                        st.error(message)

    # Only show the rest of the interface if credentials are verified
    if st.session_state.credentials_submitted:
        # Question input
        user_question = st.text_input("Ask a question (works with or without uploaded files)")

        if user_question:
            with st.spinner('Getting response...'):
                try:
                    response = get_response_from_ibm(
                        user_question, 
                        st.session_state.api_key, 
                        st.session_state.deployment_id
                    )
                    st.write(response)
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")

if __name__ == "__main__":
    main()
