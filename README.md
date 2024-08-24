# chatbot_rag_implement

## Overview

This FastAPI application provides a question-answering system leveraging Chroma for document storage and LangChain for natural language processing. The system can insert question-answer pairs into a database and search for answers to given questions based on the context of stored documents.

## Features

- **Upsert Endpoint**: Add new question-answer pairs to the database.
- **Answer Endpoint**: Retrieve answers to questions based on context from the database.
- **Reset Database Endpoint**: Reset and initialize the question-answer database from CSV files.

## Endpoints

- **`POST /upsert`**
  - **Description**: Inserts a new question-answer pair into the specified collection.
  - **Request Body**: 
    ```json
    {
      "q_a": "string",
      "collection_name": "string"
    }
    ```
  - **Response**: 
    ```json
    {
      "message": "Document upserted successfully"
    }
    ```
  - **Error**: 
    ```json
    {
      "detail": "error_message"
    }
    ```

- **`POST /answer`**
  - **Description**: Returns an answer to the given question.
  - **Request Body**:
    ```json
    {
      "question": "string",
      "prompt": "string (optional)"
    }
    ```
  - **Response**:
    ```json
    {
      "answer": "string"
    }
    ```
  - **Error**:
    ```json
    {
      "detail": "error_message"
    }
    ```

- **`POST /reset_db`**
  - **Description**: Resets the database by loading question-answer pairs from CSV files.
  - **Response**:
    ```json
    {
      "message": "Database reset successfully"
    }
    ```
  - **Error**:
    ```json
    {
      "detail": "error_message"
    }
    ```

## Deployment on GCP with Ubuntu

### Prerequisites

1. **Google Cloud Account**: Ensure you have an active Google Cloud account.
2. **gcloud CLI**: Install and initialize the Google Cloud SDK on your local machine.
3. **Ubuntu Server**: Set up an Ubuntu server on Google Cloud.

### Steps

1. **Create a Virtual Machine Instance**:
    - Open the [Google Cloud Console](https://console.cloud.google.com/).
    - Navigate to `Compute Engine` > `VM instances`.
    - Click `Create Instance`.
    - Configure your instance (e.g., select machine type, Ubuntu OS).
    - Click `Create`.

2. **SSH into Your Instance**:
    - Use the Google Cloud Console or your terminal:
      ```sh
      gcloud compute ssh <instance-name>
      ```

3. **Install Dependencies**:
    - Update packages and install necessary dependencies:
      ```sh
      sudo apt update
      sudo apt install python3-pip -y
      ```

4. **Clone the Repository**:
    - Navigate to your desired directory and clone the repository:
      ```sh
      git clone https://github.com/AdminEHR/EHR-ChatBot-Repo
      cd chatbot
      ```

5. **Install Python Packages**:
    - Install required Python packages:
      ```sh
      pip3 install -r requirements.txt
      ```
6. **Change the PORT (8000) Rules**:
    - Change the PORT permissions for API deployment:
      ```sh
      sudo ufw allow 8000
      ```

7. **Run the Application**:
    - Start the FastAPI application using Uvicorn:
      ```sh
      nohup python fast_api_blood.py
      ```

8. **Configure Firewall Rules**:
    - Allow traffic to port 8000 in the Google Cloud Console:
      - Navigate to `VPC network` > `Firewall rules`.
      - Click `Create firewall rule`.
      - Set targets, source IP ranges, and specify `tcp:8000` under `Protocols and ports`.
      - Click `Create`.

9. **Access the API**:
    - The API will be accessible at `http://<external-ip>:8000`.

## Conclusion

You have successfully deployed a FastAPI question-answering system on a Google Cloud Ubuntu server. You can now upsert documents, retrieve answers, and reset the database using the provided endpoints.
