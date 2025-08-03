#!/usr/bin/env python3
"""
Quick demo runner that sets up sample data and runs the Streamlit app
"""

import os
import sys
import subprocess
from utils import generate_sample_data
import json

def setup_demo():
    """Set up demo data and run the application"""
    print("🚀 Setting up E-commerce RAG Demo...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Check if data already exists
    if not os.path.exists("data/demo_products.json"):
        print("📦 Generating sample product data...")
        sample_products = generate_sample_data()
        
        with open("data/demo_products.json", "w", encoding="utf-8") as f:
            json.dump(sample_products, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created sample data with {len(sample_products)} products")
    
    print("🌟 Starting Streamlit application...")
    print("📱 Open your browser to: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application")
    
    # Run Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    setup_demo()