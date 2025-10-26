# HCP Darija Chatbot: A RAG-Powered Assistant for Moroccan Socio-Economic Data

This project is an advanced RAG (Retrieval-Augmented Generation) chatbot designed to bridge the gap between complex socio-economic data and the Moroccan public. By leveraging a powerful AI stack, this chatbot can understand and answer questions in **Darija (Moroccan Arabic)**, providing insights directly from **Human Capital Project (HCP)** data.

The core of the project is a Retrieval-Augmented Generation (RAG) architecture that ensures answers are accurate, context-aware, and grounded in the provided data, minimizing the risk of AI hallucinations.

## ✨ Key Features

-   **Natural Language Interaction in Darija**: Users can ask complex questions in Moroccan Arabic, making critical data accessible to a wider audience.
-   **Retrieval-Augmented Generation (RAG)**: Provides accurate, fact-based answers by retrieving relevant information before generating a response.
-   **High-Speed Vector Search**: Powered by **FAISS (Facebook AI Similarity Search)** for instant, efficient retrieval of the most relevant data passages.
-   **Advanced AI Model**: Utilizes the **Google Gemini API** for state-of-the-art natural language understanding and generation.
-   **Specialized Knowledge Base**: Integrated with **Human Capital Project (HCP)** data to provide specific socio-economic insights and analysis.

## ⚙️ How It Works (Architecture)

The chatbot follows a sophisticated RAG workflow to ensure high-quality responses:

1.  **User Query**: A user asks a question in Darija (e.g., "Chno homa l'ahdaf dyal l'proje dyal HCP?").
2.  **Vector Embedding**: The user's question is converted into a numerical representation (vector embedding) using a `sentence-transformers` model.
3.  **Similarity Search**: **FAISS** instantly searches its pre-built index of HCP data vectors to find the most semantically similar and relevant text chunks.
4.  **Context-Augmented Prompt**: The original question and the retrieved text chunks are combined into a rich prompt.
5.  **AI Generation**: This augmented prompt is sent to the **Gemini API**, which generates a comprehensive and accurate answer in Darija, based *only* on the provided context.

## 🛠️ Tech Stack

-   **Backend**: Flask
-   **AI Model**: Google Gemini API
-   **Vector Indexing**: FAISS (Facebook AI Similarity Search)
-   **Embeddings**: Sentence-Transformers
-   **Frontend**: HTML, CSS, JavaScript

## 🚀 Getting Started

Follow these steps to get the backend server running locally.

### 1. Prerequisites

-   Python 3.9+
-   Git and Git LFS (for handling large data files)

### 2. Clone the Repository

Clone the project to your local machine. Git LFS will automatically download the large data files.

```bash
git clone https://github.com/your-username/chatbot-hcp-project.git
cd chatbot-hcp-project/backend
```

### 3. Create a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

1.  In the `backend` directory, create a new file named `.env`.
2.  Add your Google Gemini API key to this file:

    ```bash    GOOGLE_API_KEY="YOUR_SECRET_API_KEY_HERE"
    ```

### 6. Run the Application

Start the Flask development server.

```bash
python main.py
```

The server will now be running. To interact with the chatbot, open the `index.html` file from the `frontend` folder in your web browser.
