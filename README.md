
# Text Analysis Pipeline

The **Text Analysis Pipeline** is a tool that analyzes text input to provide insights such as category classification, entity extraction, text summarization, keyword identification, and context determination. It's built using **Streamlit** for the user interface and uses the **Groq API** to perform language processing tasks. **OOPENAI API** 
is used for text summarization.



## Features

- **Text Classification**: Determines the category of the text (e.g., News, Blog, Research).
- **Entity Extraction**: Identifies people, organizations, dates, and other important entities.
- **Text Summarization**: Produces a concise summary of the text input.
- **Keyword Identification**: Extracts important keywords.
- **Context Identification**: Determines the context or theme (e.g., Business, Technology).

## Tech

- **Python**: Programming language
- **Streamlit**: User interface for text input and output
- **Langchain and LangGraph**: Framework for creating the text analysis pipeline
- **Groq API**: Provides the language model for NLP tasks

## Installation

### Prerequisites

Before setting up the project, make sure you have:
- **Python 3.8 or higher** installed on your system.
- **Groq API Key**
- **OOPENAI API Key**

### Steps 
1. **Install the required Python packages**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Add your Groq API Key**:
   - Create a `.env` file in the root of the project folder.
   - Add the following line to the file:
     ```bash
     GROQ_API_KEY=your_groq_api_key
     ```
2. **Add your OPENAI_API_KEY**:
   - Create a `.env` file in the root of the project folder.
   - Add the following line to the file:
     ```bash
     OPENAI_API_KEY=your_openai_api_key
     ```

4. **Run the application**:
    ```bash
    streamlit run your_filename.py
    ```

5. **Open the app**: 
   - The app will start in your browser at `http://localhost:8501`.

##Usage

1. **Input Text**: 
   - Open the app and enter your text in the text area.
   
2. **Start Analysis**:
   - Click the "Start Analysis" button. The tool will process your text and display:
     - **Category**: The category the text belongs to (e.g., News, Blog, etc).
     - **Entities**: List of entities such as people, places, or dates.
     - **Summary**: A brief summary of the text.
     - **Keywords**: Important keywords from the text.
     - **Context**: The overall context of the text (e.g., Business, Technology, etc.).

##Project Structure

```
text-analysis-pipeline/          
├── text-analysis.py     # Main file for Streamlit app
├── requirements.txt     # Dependencies required to run the project
├── README.md            # Project documentation (this file)
└── .env.example         # Example file for environment variables (e.g.,OPENAI_API_KEY, GROQ_API_KEY)
```
