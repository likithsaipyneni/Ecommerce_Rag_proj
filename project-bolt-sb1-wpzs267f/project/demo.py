"""
Demo script to quickly test the RAG system functionality
Run with: python demo.py
"""

from rag import RAGSystem
from utils import load_product_data
import json

def main():
    print("🤖 Initializing RAG System...")
    rag_system = RAGSystem()
    
    print("📦 Loading product data...")
    products = load_product_data()
    print(f"Loaded {len(products)} products")
    
    if not products:
        print("❌ No product data found. Please ensure data/*.json files exist.")
        return
    
    # Index products
    print("🔍 Indexing products...")
    rag_system.index_products(products)
    
    # Test queries
    test_queries = [
        "wireless headphones with good battery life",
        "gaming laptop with RTX graphics",
        "fitness equipment for home gym",
        "coffee machine for espresso"
    ]
    
    print("\n🎯 Testing recommendations...")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        recommendations = rag_system.get_recommendations(query, products, max_results=3)
        
        for i, (product, score) in enumerate(recommendations):
            print(f"  {i+1}. {product['title']} - Score: {score:.3f}")
    
    print("\n✅ Demo completed successfully!")
    print("Run 'streamlit run app.py' to start the full application.")

if __name__ == "__main__":
    main()