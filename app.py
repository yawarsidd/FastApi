import streamlit as st
import requests

# FastAPI backend URL
BASE_URL = "http://127.0.0.1:8000"

st.title("PDF Question and Answer System with FastAPI")

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.info("PDF is being uploaded...")
    
    # Upload the file to FastAPI backend
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    response = requests.post(f"{BASE_URL}/upload_pdf/", files=files)
    
    if response.status_code == 200:
        st.success("PDF uploaded and processed successfully")
    else:
        st.error(f"Error uploading PDF: {response.json()['detail']}")

# Ask a question
question = st.text_input("Ask a question based on the uploaded PDF:")

if st.button("Submit"):
    if not question:
        st.error("Please enter a question.")
    else:
        # Make request to FastAPI backend
        question_data = {"input": question}
        response = requests.post(f"{BASE_URL}/ask_question/", json=question_data)
        
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
            st.write(f"Answer: {answer}")
        else:
            st.error(f"Error: {response.json()['detail']}")