import json
import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from llama_cpp import Llama
from typing import List, Dict, Any, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import time

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)

# Initialize Mistral model
@st.cache_resource
def load_model():
    return Llama(
        model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx=4096,
        n_threads=4,  # Adjust based on your CPU
        n_batch=512
    )

# Initialize sentence transformer for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to process transcript data
def process_transcript(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    transcript = data.get('call_transcript', [])
    metadata = {
        'call_ID': data.get('call_ID', ''),
        'CSR_ID': data.get('CSR_ID', ''),
        'call_date': data.get('call_date', ''),
        'call_time': data.get('call_time', '')
    }
    
    questions = []
    answers = []
    greeting = ""
    
    # Process transcript
    for i, entry in enumerate(transcript):
        if entry.startswith("CSR:"):
            csr_text = entry[4:].strip()
            if i == 0:  # First CSR entry is greeting
                greeting = csr_text
            else:
                answers.append(csr_text)
        elif entry.startswith("Customer:"):
            questions.append(entry[9:].strip())
    
    # Ensure one-to-one mapping
    min_len = min(len(questions), len(answers))
    questions = questions[:min_len]
    answers = answers[:min_len]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Questions': questions,
        'Answers': answers,
        'call_ID': metadata['call_ID'],
        'CSR_ID': metadata['CSR_ID'],
        'call_date': metadata['call_date'],
        'call_time': metadata['call_time'],
        'Greeting': greeting
    })
    
    return df

# Function to analyze sentence length
def analyze_sentence_length(text: str) -> Tuple[bool, str]:
    sentences = re.split(r'[.!?]+', text)
    long_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 15]
    
    if long_sentences:
        feedback = f"Found {len(long_sentences)} long sentences. Consider breaking them down into shorter, more concise statements. "
        feedback += "Long sentences increase Average Handling Time as they take longer to deliver and may confuse customers."
        return True, feedback
    else:
        return False, "Good job using concise sentences, which helps reduce Average Handling Time."

# Function to detect repetition and crutch words
def analyze_repetition_and_crutch(text: str) -> Tuple[bool, str]:
    # Common crutch words
    crutch_words = ["um", "uh", "like", "you know", "sort of", "kind of", "basically", "actually", "literally", "honestly", "just"]
    
    # Count crutch words
    crutch_count = sum(len(re.findall(r'\\b' + word + r'\\b', text.lower())) for word in crutch_words)
    
    # Check for phrase repetition
    words = text.lower().split()
    repeated_phrases = []
    
    for i in range(len(words) - 2):
        phrase = ' '.join(words[i:i+3])
        rest_of_text = ' '.join(words[i+3:])
        if phrase in rest_of_text:
            repeated_phrases.append(phrase)
    
    repeated_phrases = list(set(repeated_phrases))
    
    if crutch_count > 0 or repeated_phrases:
        feedback = ""
        if crutch_count > 0:
            feedback += f"Found {crutch_count} instances of crutch words. "
        if repeated_phrases:
            feedback += f"Found repeated phrases: {', '.join(repeated_phrases)}. "
        
        feedback += "Repetition and crutch words increase Average Handling Time unnecessarily and may make you sound less confident."
        return True, feedback
    else:
        return False, "Good job avoiding repetition and crutch words, which helps maintain efficient call handling."

# Function to detect hold requests
def analyze_hold_requests(text: str) -> Tuple[bool, str]:
    hold_patterns = [
        r'hold\s+(?:on|please)',
        r'(?:please\s+)?wait\s+(?:a\s+)?(?:moment|second|minute)',
        r'(?:let|allow)\s+me\s+(?:to\s+)?check',
        r'one\s+(?:moment|second|minute)',
        r'bear\s+with\s+me'
    ]
    
    for pattern in hold_patterns:
        if re.search(pattern, text.lower()):
            return True, "Detected request for customer to hold or wait. This increases Average Handling Time. Consider if you can prepare information in advance or use more efficient systems navigation."
    
    return False, "No hold requests detected, which is good for maintaining efficient Average Handling Time."

# Function to detect call transfers
def analyze_call_transfers(text: str) -> Tuple[bool, str, str]:
    transfer_patterns = [
        r'(?:transfer|transferring)\s+(?:you|your\s+call)',
        r'(?:connect|connecting)\s+(?:you|your\s+call)',
        r'(?:let\s+me\s+get|getting)\s+(?:you|your\s+call)\s+to',
        r'(?:put|putting)\s+(?:you|your\s+call)\s+through',
        r'(?:hand|handing)\s+(?:you|your\s+call)\s+over'
    ]
    
    for pattern in transfer_patterns:
        if re.search(pattern, text.lower()):
            # Use Mistral to identify reason for transfer
            model = load_model()
            prompt = f"""
            Analyze the following customer service representative statement and identify the reason for transferring the call:
            
            "{text}"
            
            What is the specific reason for transferring this call? Provide a brief explanation.
            """
            
            response = model(prompt, max_tokens=100, temperature=0.1)
            reason = response['choices'][0]['text'].strip()
            
            return True, f"Detected call transfer. This may indicate a training opportunity to handle more issues without transfers.", reason
    
    return False, "No call transfers detected, which is good for first-contact resolution.", ""

# Function to create knowledge documents
def create_knowledge_documents(df: pd.DataFrame) -> List[str]:
    documents = []
    
    # Create documents from Q&A pairs
    for _, row in df.iterrows():
        doc = f"Question: {row['Questions']}\nAnswer: {row['Answers']}\n"
        documents.append(doc)
    
    # Add metadata context
    metadata_doc = f"Call ID: {df['call_ID'].iloc[0]}\nCSR ID: {df['CSR_ID'].iloc[0]}\nDate: {df['call_date'].iloc[0]}\nTime: {df['call_time'].iloc[0]}"
    documents.append(metadata_doc)
    
    return documents

# Function to chunk documents
def chunk_documents(documents: List[str], chunk_size: int = 200) -> List[str]:
    chunks = []
    
    for doc in documents:
        words = doc.split()
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
    
    return chunks

# RAG Pipeline
class RAGPipeline:
    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.chunks = []
        self.index = None
        self.llm = load_model()
    
    def add_documents(self, documents: List[str]):
        self.chunks = chunk_documents(documents)
        self._build_index()
    
    def _build_index(self):
        # Create embeddings
        embeddings = self.embedding_model.encode(self.chunks)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store embeddings
        self.embeddings = embeddings
    
    def query(self, question: str, k: int = 3) -> str:
        # Get question embedding
        question_embedding = self.embedding_model.encode([question])[0]
        
        # Search index
        distances, indices = self.index.search(np.array([question_embedding]).astype('float32'), k)
        
        # Get relevant chunks
        relevant_chunks = [self.chunks[i] for i in indices[0]]
        context = "\n".join(relevant_chunks)
        
        # Generate answer with Mistral
        prompt = f"""
        You are an expert customer service representative. Use the following context to answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a professional, helpful, and concise answer:
        """
        
        response = self.llm(prompt, max_tokens=300, temperature=0.1)
        return response['choices'][0]['text'].strip()

# Custom evaluation function using Mistral
def evaluate_answer_relevancy(question: str, answer: str, model) -> float:
    prompt = f"""
    Evaluate how relevant the following answer is to the question on a scale from 0.0 to 1.0.
    
    Question: {question}
    Answer: {answer}
    
    A score of 1.0 means the answer is perfectly relevant and addresses all aspects of the question.
    A score of 0.0 means the answer is completely irrelevant to the question.
    
    Provide only a numerical score between 0.0 and 1.0:
    """
    
    response = model(prompt, max_tokens=10, temperature=0.1)
    response_text = response['choices'][0]['text'].strip()
    
    # Extract the numerical score
    try:
        score = float(re.search(r'(\d+\.\d+|\d+)', response_text).group(1))
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        return score
    except (AttributeError, ValueError):
        # Default to middle score if parsing fails
        return 0.5

def evaluate_answer_faithfulness(answer: str, expected_answer: str, model) -> float:
    prompt = f"""
    Evaluate how factually accurate the following answer is compared to the reference answer on a scale from 0.0 to 1.0.
    
    Answer to evaluate: {answer}
    Reference answer: {expected_answer}
    
    A score of 1.0 means the answer is completely factually accurate and contains no incorrect information.
    A score of 0.0 means the answer contains significant factual errors or misinformation.
    
    Provide only a numerical score between 0.0 and 1.0:
    """
    
    response = model(prompt, max_tokens=10, temperature=0.1)
    response_text = response['choices'][0]['text'].strip()
    
    # Extract the numerical score
    try:
        score = float(re.search(r'(\d+\.\d+|\d+)', response_text).group(1))
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        return score
    except (AttributeError, ValueError):
        # Default to middle score if parsing fails
        return 0.5

# Function to evaluate CSR answers
def evaluate_answers(df: pd.DataFrame, rag_answers: List[str]) -> List[Dict[str, Any]]:
    results = []
    
    # Load model for evaluation
    model = load_model()
    
    for i, row in df.iterrows():
        question = row['Questions']
        csr_answer = row['Answers']
        rag_answer = rag_answers[i]
        
        # Evaluate relevancy and faithfulness
        relevancy_score = evaluate_answer_relevancy(question, csr_answer, model)
        faithfulness_score = evaluate_answer_faithfulness(csr_answer, rag_answer, model)
        
        # Generate feedback
        relevancy_feedback = f"Your Relevancy score is {relevancy_score:.2f}. "
        if relevancy_score < 0.7:
            relevancy_feedback += "Your answer didn't fully address the customer's question. "
            relevancy_feedback += f"You should focus more on addressing: {question}"
        else:
            relevancy_feedback += "You did a good job addressing the customer's main concerns."
        
        faithfulness_feedback = f"Your Faithfulness score is {faithfulness_score:.2f}. "
        if faithfulness_score < 0.7:
            faithfulness_feedback += "Your answer contained some inaccurate information. "
            faithfulness_feedback += "You need to brush up on your product knowledge and procedures."
        else:
            faithfulness_feedback += "Your information was accurate and helpful."
        
        results.append({
            'question': question,
            'csr_answer': csr_answer,
            'rag_answer': rag_answer,
            'relevancy_score': relevancy_score,
            'faithfulness_score': faithfulness_score,
            'relevancy_feedback': relevancy_feedback,
            'faithfulness_feedback': faithfulness_feedback
        })
    
    return results

# Function to analyze sentiment
def analyze_sentiment(text: str) -> Tuple[float, str]:
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    compound = sentiment['compound']
    
    if compound >= 0.5:
        feedback = "You were very positive and enthusiastic in your response, which helps build rapport with customers."
    elif compound >= 0.1:
        feedback = "You maintained a positive tone in your response, which is good for customer satisfaction."
    elif compound > -0.1:
        feedback = "Your tone was neutral. Consider adding more warmth and positivity to build better customer rapport."
    elif compound > -0.5:
        feedback = "Your tone came across as somewhat negative. Try to maintain a more positive and supportive tone."
    else:
        feedback = "Your tone was very negative. It's important to maintain a positive and supportive tone even in difficult situations."
    
    return compound, feedback

# Function to summarize topic
def summarize_topic(question: str) -> str:
    model = load_model()
    
    prompt = f"""
    Summarize the main topic or theme of the following customer question in a brief phrase (5-10 words):
    
    "{question}"
    
    Topic/Theme:
    """
    
    response = model(prompt, max_tokens=50, temperature=0.1)
    return response['choices'][0]['text'].strip()

# Function to create concise summary
def create_concise_summary(feedback_data: Dict[str, Any]) -> str:
    model = load_model()
    
    # Compile all feedback
    all_feedback = ""
    
    if 'sentence_length' in feedback_data:
        all_feedback += feedback_data['sentence_length'] + "\n\n"
    
    if 'repetition' in feedback_data:
        all_feedback += feedback_data['repetition'] + "\n\n"
    
    if 'hold_requests' in feedback_data:
        all_feedback += feedback_data['hold_requests'] + "\n\n"
    
    if 'transfers' in feedback_data:
        all_feedback += feedback_data['transfers'] + "\n\n"
        if 'transfer_reason' in feedback_data and feedback_data['transfer_reason']:
            all_feedback += "Transfer reason: " + feedback_data['transfer_reason'] + "\n\n"
    
    if 'evaluation_results' in feedback_data:
        for result in feedback_data['evaluation_results']:
            all_feedback += result['relevancy_feedback'] + "\n"
            all_feedback += result['faithfulness_feedback'] + "\n\n"
    
    # Generate concise summary
    prompt = f"""
    Summarize the following customer service representative feedback in approximately 300 words.
    Focus on the most important points and actionable improvements.
    
    Feedback:
    {all_feedback}
    
    Concise Summary (300 words):
    """
    
    response = model(prompt, max_tokens=400, temperature=0.1)
    return response['choices'][0]['text'].strip()

# Main Streamlit app
def main():
    st.title("Call Center Performance Coach")
    st.subheader("Powered by Mistral 7B")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload call transcript JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_transcript.json", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process transcript
        with st.spinner("Processing transcript..."):
            df = process_transcript("temp_transcript.json")
            
            # Display basic info
            st.subheader("Call Information")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Call ID", df['call_ID'].iloc[0])
            col2.metric("CSR ID", df['CSR_ID'].iloc[0])
            col3.metric("Date", df['call_date'].iloc[0])
            col4.metric("Time", df['call_time'].iloc[0])
            
            # Display greeting
            st.subheader("Greeting")
            st.write(df['Greeting'].iloc[0])
            
            # Display Q&A
            st.subheader("Conversation")
            for i, row in df.iterrows():
                st.markdown(f"**Customer:** {row['Questions']}")
                st.markdown(f"**CSR:** {row['Answers']}")
                st.markdown("---")
        
        # Run evaluation button
        if st.button("Run Mistral Evaluation"):
            with st.spinner("Running evaluation... This may take a few minutes"):
                # Initialize progress
                progress_bar = st.progress(0)
                
                # Step 1: Analyze sentence length
                progress_bar.progress(10)
                all_csr_text = " ".join(df['Answers'].tolist()) + " " + df['Greeting'].iloc[0]
                has_long_sentences, sentence_feedback = analyze_sentence_length(all_csr_text)
                
                # Step 2: Analyze repetition and crutch words
                progress_bar.progress(20)
                has_repetition, repetition_feedback = analyze_repetition_and_crutch(all_csr_text)
                
                # Step 3: Analyze hold requests
                progress_bar.progress(30)
                has_hold_requests, hold_feedback = analyze_hold_requests(all_csr_text)
                
                # Step 4: Analyze transfers
                progress_bar.progress(40)
                has_transfers, transfer_feedback, transfer_reason = analyze_call_transfers(all_csr_text)
                
                # Step 5: Create knowledge documents
                progress_bar.progress(50)
                documents = create_knowledge_documents(df)
                
                # Step 6: Build RAG pipeline
                progress_bar.progress(60)
                rag = RAGPipeline()
                rag.add_documents(documents)
                
                # Step 7: Generate RAG answers
                progress_bar.progress(70)
                rag_answers = []
                for question in df['Questions']:
                    rag_answers.append(rag.query(question))
                
                # Step 8: Evaluate with custom evaluation
                progress_bar.progress(80)
                evaluation_results = evaluate_answers(df, rag_answers)
                
                # Step 9: Analyze sentiment
                progress_bar.progress(90)
                sentiment_scores = []
                sentiment_feedback = []
                for answer in df['Answers']:
                    score, feedback = analyze_sentiment(answer)
                    sentiment_scores.append(score)
                    sentiment_feedback.append(feedback)
                
                # Step 10: Summarize topics
                topics = []
                for question in df['Questions']:
                    topics.append(summarize_topic(question))
                
                # Step 11: Create concise summary
                feedback_data = {
                    'sentence_length': sentence_feedback,
                    'repetition': repetition_feedback,
                    'hold_requests': hold_feedback,
                    'transfers': transfer_feedback,
                    'transfer_reason': transfer_reason,
                    'evaluation_results': evaluation_results
                }
                
                concise_summary = create_concise_summary(feedback_data)
                
                # Complete progress
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                
                # Display results
                st.subheader("Evaluation Results")
                
                # Display tabs for different feedback categories
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "Summary", "Sentence Structure", "Repetition & Crutch Words", 
                    "Hold Requests", "Transfers", "Answer Quality"
                ])
                
                with tab1:
                    st.subheader("Concise Summary")
                    st.write(concise_summary)
                    
                    st.subheader("CSR Performance Overview")
                    avg_relevancy = sum(r['relevancy_score'] for r in evaluation_results) / len(evaluation_results)
                    avg_faithfulness = sum(r['faithfulness_score'] for r in evaluation_results) / len(evaluation_results)
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg. Relevancy", f"{avg_relevancy:.2f}")
                    col2.metric("Avg. Faithfulness", f"{avg_faithfulness:.2f}")
                    col3.metric("Avg. Sentiment", f"{avg_sentiment:.2f}")
                
                with tab2:
                    st.subheader("Sentence Structure Analysis")
                    st.write(sentence_feedback)
                    if has_long_sentences:
                        st.warning("Long sentences detected")
                    else:
                        st.success("Good sentence structure")
                
                with tab3:
                    st.subheader("Repetition & Crutch Words Analysis")
                    st.write(repetition_feedback)
                    if has_repetition:
                        st.warning("Repetition or crutch words detected")
                    else:
                        st.success("Good language usage")
                
                with tab4:
                    st.subheader("Hold Requests Analysis")
                    st.write(hold_feedback)
                    if has_hold_requests:
                        st.warning("Hold requests detected")
                    else:
                        st.success("No hold requests detected")
                
                with tab5:
                    st.subheader("Transfers Analysis")
                    st.write(transfer_feedback)
                    if has_transfers:
                        st.warning("Call transfer detected")
                        st.write(f"**Transfer reason:** {transfer_reason}")
                    else:
                        st.success("No transfers detected")
                
                with tab6:
                    st.subheader("Answer Quality Evaluation")
                    for i, result in enumerate(evaluation_results):
                        with st.expander(f"Question {i+1}: {topics[i]}"):
                            st.write("**Customer Question:**")
                            st.write(result['question'])
                            
                            st.write("**CSR Answer:**")
                            st.write(result['csr_answer'])
                            
                            st.write("**Ideal Answer (RAG):**")
                            st.write(result['rag_answer'])
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Relevancy Score", f"{result['relevancy_score']:.2f}")
                            col2.metric("Faithfulness Score", f"{result['faithfulness_score']:.2f}")
                            
                            st.write("**Relevancy Feedback:**")
                            st.write(result['relevancy_feedback'])
                            
                            st.write("**Faithfulness Feedback:**")
                            st.write(result['faithfulness_feedback'])
                            
                            st.write("**Sentiment Analysis:**")
                            st.write(sentiment_feedback[i])
                
                # Clean up
                if os.path.exists("temp_transcript.json"):
                    os.remove("temp_transcript.json")

if __name__ == "__main__":
    main()

