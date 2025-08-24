# Call Center Performance Coach

A Python application that uses the Mistral 7B model to evaluate call center transcripts and provide detailed feedback for customer service representatives.

## Features

- Analyzes call center transcripts in JSON format
- Identifies areas for improvement in CSR responses:
  - Long sentences
  - Repetition and crutch words
  - Hold requests
  - Call transfers
- Builds a RAG (Retrieval-Augmented Generation) pipeline for ideal answers
- Evaluates CSR answers using Deepeval metrics
- Provides sentiment analysis
- Summarizes topics and themes
- Creates concise feedback summaries
- Streamlit frontend for easy interaction

## Requirements

- Python 3.8+
- Mistral 7B model (mistral-7b-instruct-v0.2.Q4_K_M.gguf)
- CPU-based environment (no GPU required)
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place the Mistral 7B model file (mistral-7b-instruct-v0.2.Q4_K_M.gguf) in the root directory

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Upload a call transcript JSON file
3. Click "Run Mistral Evaluation" to analyze the transcript
4. View the detailed feedback and metrics

## Input Format

The application expects call transcripts in JSON format with the following structure:

```json
{
  "call_transcript": [
    "CSR: Thank you for calling...",
    "Customer: I need help with...",
    "CSR: I can help you with that...",
    ...
  ],
  "call_ID": "12345",
  "CSR_ID": "JSmith",
  "call_date": "2024-02-06",
  "call_time": "21:58:05"
}
```

## Output

The application provides:
- Sentence structure analysis
- Repetition and crutch word detection
- Hold request identification
- Transfer analysis
- Answer quality evaluation using RAG and Deepeval
- Sentiment analysis
- Topic summarization
- Concise feedback summary

