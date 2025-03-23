# Hotel Booking RAG System

A Retrieval-Augmented Generation (RAG) system for querying hotel booking data using natural language. This project combines vector search with the Gemini AI model to provide intelligent answers to questions about hotel booking analytics.

## Overview

This system processes hotel booking data to generate comprehensive analytics and allows users to query the data through a natural language interface. The architecture combines:

1. **Data Preprocessing**: Transforms raw hotel booking data into structured analytics
2. **Vector Search**: Uses FAISS and HuggingFace embeddings to create a searchable knowledge base
3. **LLM Integration**: Leverages Google's Gemini AI model to generate human-like responses
4. **API Access**: Provides a Flask API for integration with other applications

## Features

- Interactive command-line RAG chatbot
- REST API for programmatic access
- Comprehensive hotel booking analytics
- Natural language querying of hotel data
- Fast vector search with FAISS

## Requirements

- Python 3.8+
- FAISS
- LangChain
- Hugging Face Transformers
- Google Generative AI
- Flask
- pandas
- numpy
- tqdm

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hotel-booking-rag.git
cd hotel-booking-rag
```

### 2. Create and Activate Virtual Environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Using requirements.txt:
```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install langchain langchain-community langchain-huggingface langchain-google-genai langchain-core faiss-cpu sentence-transformers google-generativeai python-dotenv flask pandas numpy tqdm ipython
```

### 4. Setup Environment Variables

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key
```

### 5. Prepare Data

Place your `hotel_bookings.csv` file in the project root directory.

## Running the Application

### Option 1: Interactive Command-Line Chatbot

To chat using the full interactive experience with progress bars:
```bash
python rag.py
```

For a simpler command-line interface:
```bash
python question.py
```

### Option 2: Web API

Start the Flask server:
```bash
python app.py
```

Access the API endpoints at http://localhost:5000/

## API Endpoints



### 1. Ask Question
- **URL**: `/ask`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "question": "What is the cancellation rate for Resort Hotel?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "The cancellation rate for Resort Hotel is 27.76%."
  }
  ```

### 2. Get Analytics
- **URL**: `/analytics`
- **Method**: `POST`
- **Response**: Full analytics report as JSON

## Project Structure

- `rag.py`: Main RAG implementation with vector store creation and chatbot setup
- `rag_preprocessing.py`: Data preprocessing and insights generation
- `analytical_report.py`: Detailed analytics calculations
- `question.py`: Question-answering class for easy integration
- `app.py`: Flask API for web access
- `hotel_bookings.csv`: Raw data (not included in repository)

## Sample Queries and Results

Here are some example queries you can make to the API and their expected results:

### Example 1: Revenue for a Specific Month

```bash
curl -X POST http://127.0.0.1:5000/ask \
-H "Content-Type: application/json" \
-d '{"question": "Show me total revenue for July 2017."}'
```

Response:
```json
{"answer":"The total revenue for July 2017 was $1,817,038.23."}
```

### Example 2: Booking Cancellations

```bash
curl -X POST http://127.0.0.1:5000/ask \
-H "Content-Type: application/json" \
-d '{"question": "Which locations had the highest booking cancellations?"}'
```

Response:
```json
{"answer":"City Hotel had the highest booking cancellation rate at 41.73%."}
```

### Example 3: Average Price

```bash
curl -X POST http://127.0.0.1:5000/ask \
-H "Content-Type: application/json" \
-d '{"question": "What is the average price of a hotel booking?"}'
```

Response:
```json
{"answer":"The average price of a hotel booking is $101.83."}
```

### Example 4: Retrieving Full Analytics

```bash
curl -X POST http://127.0.0.1:5000/analytics
```

Response:
A comprehensive JSON with all analytics data including:
- Revenue analysis
- Cancellation analysis
- Geographical analysis
- Lead time analysis
- Additional insights

## Development

This project is structured with modular components that work together to provide RAG capabilities:

### Data Processing Pipeline

The data pipeline consists of three main components:

1. **analytical_report.py**:
   - Contains functions for generating detailed analytics on hotel booking data
   - Calculates revenue trends, cancellation rates, geographical distribution, lead times, etc.
   - Functions like `calculate_revenue_trends()`, `calculate_cancellations()`, and `calculate_geography()` process different aspects of the data
   - The `generate_report()` function combines all analytics into a single report

2. **rag_preprocessing.py**:
   - Builds on the analytical framework to create structured data for the RAG system
   - Uses functions like `calculate_overall_insights()` and `calculate_hotel_insights()` to summarize data
   - The `generate_data()` function combines insights with the analytical report
   - Handles complex data types through custom JSON serialization

### RAG Implementation

The core RAG functionality is implemented in `rag.py`:

1. **Vectorization**:
   - Converts dictionary data to LangChain Document objects
   - Splits documents into chunks for better retrieval
   - Uses HuggingFace embeddings to convert text into vector representations
   - Creates a FAISS vector store for efficient similarity search

2. **Prompt Engineering**:
   - Defines a template that combines retrieved context with user questions
   - Structures the prompt to guide the LLM's responses

3. **Pipeline Integration**:
   - Constructs a pipeline that connects the retriever, prompt, and LLM
   - Uses LangChain's composable components architecture

### User Interfaces

The project offers multiple ways to interact with the RAG system:

1. **Interactive CLI (rag.py)**:
   - Full-featured interactive command-line interface
   - Includes progress bars for better user experience
   - Supports Markdown rendering for IPython environments

2. **Simple CLI (question.py)**:
   - Lightweight command-line interface
   - Encapsulates RAG functionality in a reusable Question class
   - Provides a simple API for integration with other Python code

3. **Web API (app.py)**:
   - Flask-based REST API with JSON responses
   - Exposes endpoints for querying the RAG system and retrieving analytics
   - Suitable for integration with web applications and services

### Extending the System

To add new capabilities to the system:

1. **Add New Analytics**:
   - Create new functions in `analytical_report.py` to calculate additional metrics
   - Include them in the `generate_report()` function

2. **Enhance RAG Capabilities**:
   - Modify the prompt template in `rag.py` to improve response quality
   - Experiment with different embedding models or chunk sizes
   - Adjust the number of retrieved documents with the `k` parameter

3. **Improve API**:
   - Add new endpoints in `app.py` for additional functionality
   - Implement caching for better performance
   - Add authentication for secure access