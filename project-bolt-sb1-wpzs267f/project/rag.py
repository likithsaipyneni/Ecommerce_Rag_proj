import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
from utils import chunk_product_data, analyze_sentiment, format_price

class RAGSystem:
    def __init__(self):
        """Initialize RAG system with embeddings and vector database"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "product_embeddings"
        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        # Initialize collection
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def index_products(self, products: List[Dict[str, Any]]):
        """Index products into vector database"""
        documents = []
        metadatas = []
        ids = []
        
        for product in products:
            # Chunk product data
            chunks = chunk_product_data(product)
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{product['id']}_chunk_{i}"
                
                documents.append(chunk['text'])
                metadatas.append({
                    'product_id': product['id'],
                    'chunk_type': chunk['type'],
                    'title': product['title'],
                    'category': product['category'],
                    'price': product['price'],
                    'rating': product['rating']
                })
                ids.append(doc_id)
        
        # Generate embeddings and store
        if documents:
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Clear existing data for fresh indexing
            try:
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except:
                pass
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
    
    def retrieve_relevant_products(self, query: str, n_results: int = 10) -> List[Tuple[Dict, float]]:
        """Retrieve relevant products based on query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 3  # Get more to filter for unique products
        )
        
        # Group by product and calculate relevance scores
        product_scores = {}
        for i, metadata in enumerate(results['metadatas'][0]):
            product_id = metadata['product_id']
            distance = results['distances'][0][i]
            relevance_score = 1 - distance  # Convert distance to relevance
            
            if product_id not in product_scores:
                product_scores[product_id] = {
                    'max_score': relevance_score,
                    'metadata': metadata
                }
            else:
                product_scores[product_id]['max_score'] = max(
                    product_scores[product_id]['max_score'], 
                    relevance_score
                )
        
        # Sort by relevance and return top results
        sorted_products = sorted(
            product_scores.items(),
            key=lambda x: x[1]['max_score'],
            reverse=True
        )[:n_results]
        
        return [(prod[1]['metadata'], prod[1]['max_score']) for prod in sorted_products]
    
    def get_recommendations(self, query: str, products: List[Dict], 
                          user_preferences: str = "", max_results: int = 10) -> List[Tuple[Dict, float]]:
        """Get product recommendations using RAG"""
        # Index products if collection is empty
        if self.collection.count() == 0:
            self.index_products(products)
        
        # Enhance query with user preferences
        enhanced_query = query
        if user_preferences:
            enhanced_query += f" {user_preferences}"
        
        # Retrieve relevant products
        relevant_results = self.retrieve_relevant_products(enhanced_query, max_results)
        
        # Match with full product data
        recommendations = []
        for metadata, score in relevant_results:
            product = next((p for p in products if p['id'] == metadata['product_id']), None)
            if product:
                recommendations.append((product, score))
        
        return recommendations
    
    def generate_explanation(self, query: str, recommendations: List[Tuple[Dict, float]], 
                           user_preferences: str = "") -> str:
        """Generate AI explanation for recommendations"""
        if not recommendations:
            return ""
        
        # Prepare context for LLM
        context = f"User Query: {query}\n"
        if user_preferences:
            context += f"User Preferences: {user_preferences}\n"
        
        context += "Top Recommended Products:\n"
        for i, (product, score) in enumerate(recommendations):
            context += f"{i+1}. {product['title']} - {format_price(product['price'])} (Relevance: {score:.2f})\n"
            context += f"   Category: {product['category']}, Rating: {product['rating']}/5\n"
        
        prompt = f"""Based on the user's query and the recommended products, provide a helpful explanation of why these products were recommended. Keep it concise and focus on how they match the user's needs.

{context}

Explanation:"""
        
        # Try to use HuggingFace API
        if self.hf_api_key:
            response = self._query_huggingface(prompt)
            if response:
                return response
        
        # Fallback to rule-based explanation
        return self._generate_fallback_explanation(query, recommendations, user_preferences)
    
    def compare_products(self, product1: Dict, product2: Dict) -> str:
        """Generate AI comparison between two products"""
        context = f"""
Product 1: {product1['title']}
Price: {format_price(product1['price'])}
Rating: {product1['rating']}/5
Category: {product1['category']}
Description: {product1['description'][:200]}...

Product 2: {product2['title']}
Price: {format_price(product2['price'])}
Rating: {product2['rating']}/5
Category: {product2['category']}
Description: {product2['description'][:200]}...
"""
        
        prompt = f"""Compare these two products and highlight their key differences, pros and cons. Focus on helping a customer decide between them.

{context}

Comparison Analysis:"""
        
        # Try HuggingFace API
        if self.hf_api_key:
            response = self._query_huggingface(prompt)
            if response:
                return response
        
        # Fallback comparison
        return self._generate_fallback_comparison(product1, product2)
    
    def analyze_all_sentiments(self, products: List[Dict]) -> Dict[str, int]:
        """Analyze sentiment distribution across all product reviews"""
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        
        for product in products:
            if 'reviews' in product:
                for review in product['reviews']:
                    sentiment = analyze_sentiment(review['text'])
                    sentiment_counts[sentiment] += 1
        
        return sentiment_counts
    
    def _query_huggingface(self, prompt: str, max_length: int = 200) -> Optional[str]:
        """Query HuggingFace Inference API"""
        if not self.hf_api_key:
            return None
        
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    # Extract only the new generated content
                    if prompt in generated_text:
                        return generated_text.replace(prompt, '').strip()
                    return generated_text.strip()
        except Exception as e:
            print(f"HuggingFace API error: {e}")
        
        return None
    
    def _generate_fallback_explanation(self, query: str, recommendations: List[Tuple[Dict, float]], 
                                     user_preferences: str) -> str:
        """Generate rule-based explanation when LLM is not available"""
        if not recommendations:
            return "No recommendations found for your query."
        
        explanation = f"Based on your search for '{query}', here are the top recommendations:\n\n"
        
        for i, (product, score) in enumerate(recommendations[:3]):
            explanation += f"**{i+1}. {product['title']}** (Relevance: {score:.2f})\n"
            explanation += f"- Price: {format_price(product['price'])}\n"
            explanation += f"- Rating: ⭐ {product['rating']}/5\n"
            explanation += f"- Category: {product['category']}\n"
            
            # Add specific reasons based on product features
            reasons = []
            if score > 0.8:
                reasons.append("highly relevant to your search")
            if product['rating'] >= 4.5:
                reasons.append("excellent customer ratings")
            if user_preferences and any(pref.lower() in product['description'].lower() 
                                     for pref in user_preferences.split()):
                reasons.append("matches your stated preferences")
            
            if reasons:
                explanation += f"- Recommended because: {', '.join(reasons)}\n"
            explanation += "\n"
        
        return explanation
    
    def _generate_fallback_comparison(self, product1: Dict, product2: Dict) -> str:
        """Generate rule-based comparison when LLM is not available"""
        analysis = f"**Comparison: {product1['title']} vs {product2['title']}**\n\n"
        
        # Price comparison
        if product1['price'] < product2['price']:
            price_diff = product2['price'] - product1['price']
            analysis += f"💰 **Price**: {product1['title']} is ${price_diff:.2f} cheaper\n"
        elif product1['price'] > product2['price']:
            price_diff = product1['price'] - product2['price']
            analysis += f"💰 **Price**: {product2['title']} is ${price_diff:.2f} cheaper\n"
        else:
            analysis += f"💰 **Price**: Both products are priced equally at {format_price(product1['price'])}\n"
        
        # Rating comparison
        if product1['rating'] > product2['rating']:
            analysis += f"⭐ **Rating**: {product1['title']} has higher customer satisfaction ({product1['rating']} vs {product2['rating']})\n"
        elif product1['rating'] < product2['rating']:
            analysis += f"⭐ **Rating**: {product2['title']} has higher customer satisfaction ({product2['rating']} vs {product1['rating']})\n"
        else:
            analysis += f"⭐ **Rating**: Both products have equal ratings of {product1['rating']}/5\n"
        
        # Category comparison
        if product1['category'] != product2['category']:
            analysis += f"🏷️ **Category**: Different categories - {product1['category']} vs {product2['category']}\n"
        
        # Value recommendation
        analysis += "\n**Recommendation:**\n"
        if product1['rating'] > product2['rating'] and product1['price'] <= product2['price']:
            analysis += f"✅ {product1['title']} offers better value with higher rating and equal/lower price"
        elif product2['rating'] > product1['rating'] and product2['price'] <= product1['price']:
            analysis += f"✅ {product2['title']} offers better value with higher rating and equal/lower price"
        else:
            analysis += "Both products have their merits. Consider your budget and specific feature needs."
        
        return analysis