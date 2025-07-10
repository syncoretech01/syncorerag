import os
import json
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "syncoretech-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "syncore-rag")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Flask app
app = Flask(__name__)

def preprocess_query(query):
    """Clean and enhance query for better matching"""
    # Basic cleaning
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    
    # Expand common abbreviations
    abbreviations = {
        "seo": "search engine optimization SEO",
        "ppc": "pay per click PPC advertising",
        "roi": "return on investment ROI",
        "crm": "customer relationship management CRM",
        "ui": "user interface UI",
        "ux": "user experience UX"
    }
    
    query_lower = query.lower()
    for abbr, expansion in abbreviations.items():
        if abbr in query_lower:
            query = query.replace(abbr, expansion).replace(abbr.upper(), expansion)
    
    return query

def generate_enhanced_embedding(query, context_boost=True):
    """Generate embedding with optional context enhancement"""
    processed_query = preprocess_query(query)
    
    if context_boost:
        # Add context to help with retrieval
        context_prefix = "Find information about: "
        processed_query = context_prefix + processed_query
    
    response = openai_client.embeddings.create(
        input=processed_query,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def rerank_results(query, results):
    """Rerank results based on keyword matching and relevance"""
    query_words = set(query.lower().split())
    
    for match in results:
        if 'metadata' in match and 'text' in match['metadata']:
            text_lower = match['metadata']['text'].lower()
            # Count keyword matches
            keyword_score = sum(1 for word in query_words if word in text_lower)
            # Boost exact phrase matches
            if query.lower() in text_lower:
                keyword_score += 5
            # Combine with vector score
            match['combined_score'] = match['score'] * 0.7 + (keyword_score / len(query_words)) * 0.3
    
    # Sort by combined score
    results.sort(key=lambda x: x.get('combined_score', x['score']), reverse=True)
    return results

@app.route("/query", methods=["POST"])
def query():
    """Enhanced query endpoint with better relevance"""
    data = request.get_json()
    question = data.get("question", "")
    top_k = data.get("top_k", 5)
    use_reranking = data.get("rerank", True)
    
    if not question:
        return jsonify({"error": "Question not provided"}), 400

    try:
        # Generate embedding
        query_vector = generate_enhanced_embedding(question)
        
        # Query Pinecone with higher initial k for reranking
        initial_k = top_k * 2 if use_reranking else top_k
        pinecone_response = index.query(
            vector=query_vector,
            top_k=initial_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE
        )
        
        # Convert to dict and process results
        results = pinecone_response.to_dict()
        
        if use_reranking and results.get('matches'):
            # Rerank and trim to requested top_k
            results['matches'] = rerank_results(question, results['matches'])[:top_k]
        
        # Add query info for debugging
        results['query_info'] = {
            "original_query": question,
            "processed_query": preprocess_query(question),
            "reranking_applied": use_reranking
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "index": PINECONE_INDEX_NAME})

@app.route("/query_with_context", methods=["POST"])
def query_with_context():
    """Advanced query with conversation context"""
    data = request.get_json()
    question = data.get("question", "")
    context = data.get("context", [])  # Previous conversation
    filters = data.get("filters", {})  # e.g., {"content_type": "pricing"}
    
    if not question:
        return jsonify({"error": "Question not provided"}), 400
    
    # Build context-aware query
    if context:
        context_text = " ".join(context[-3:])  # Last 3 messages
        enhanced_question = f"Context: {context_text}. Current question: {question}"
    else:
        enhanced_question = question
    
    try:
        query_vector = generate_enhanced_embedding(enhanced_question)
        
        # Build Pinecone query with filters
        query_params = {
            "vector": query_vector,
            "top_k": 5,
            "include_metadata": True,
            "namespace": PINECONE_NAMESPACE
        }
        
        if filters:
            query_params["filter"] = filters
        
        pinecone_response = index.query(**query_params)
        results = pinecone_response.to_dict()
        
        # Generate contextual answer using GPT
        if results.get('matches'):
            context_texts = [m['metadata']['text'] for m in results['matches'][:3]]
            
            prompt = f"""Based on the following information, answer the user's question.
            
Information:
{chr(10).join(context_texts)}

Question: {question}

Provide a helpful, accurate answer based solely on the given information."""
            
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            results['generated_answer'] = completion.choices[0].message.content
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)