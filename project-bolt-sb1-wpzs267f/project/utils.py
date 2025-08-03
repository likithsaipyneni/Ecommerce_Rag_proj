import re
import json
import os
from typing import List, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def chunk_product_data(product: Dict[str, Any], chunk_size: int = 200) -> List[Dict[str, str]]:
    """
    Smart chunking strategy for product data
    Creates meaningful chunks from different product attributes
    """
    chunks = []
    
    # Title and basic info chunk
    basic_info = f"Product: {product['title']}\n"
    basic_info += f"Category: {product['category']}\n"
    basic_info += f"Price: ${product['price']}\n"
    basic_info += f"Rating: {product['rating']}/5"
    
    chunks.append({
        'text': basic_info,
        'type': 'basic_info'
    })
    
    # Description chunks
    description = product.get('description', '')
    if description:
        desc_chunks = split_text_smart(description, chunk_size)
        for i, chunk in enumerate(desc_chunks):
            chunks.append({
                'text': f"Product Description: {chunk}",
                'type': f'description_{i}'
            })
    
    # Specifications chunk
    if 'specs' in product and product['specs']:
        specs_text = "Product Specifications:\n"
        for key, value in product['specs'].items():
            specs_text += f"- {key}: {value}\n"
        
        chunks.append({
            'text': specs_text,
            'type': 'specifications'
        })
    
    # Reviews chunks (group reviews by sentiment)
    if 'reviews' in product and product['reviews']:
        # Analyze sentiment for each review first
        for review in product['reviews']:
            if 'sentiment' not in review:
                review['sentiment'] = analyze_sentiment(review['text'])
        
        # Group reviews by sentiment
        positive_reviews = [r for r in product['reviews'] if r.get('sentiment') == 'Positive']
        negative_reviews = [r for r in product['reviews'] if r.get('sentiment') == 'Negative']
        neutral_reviews = [r for r in product['reviews'] if r.get('sentiment') == 'Neutral']
        
        # Create chunks for each sentiment group
        for sentiment, reviews in [('Positive', positive_reviews), ('Negative', negative_reviews), ('Neutral', neutral_reviews)]:
            if reviews:
                review_text = f"{sentiment} Customer Reviews:\n"
                for review in reviews[:3]:  # Limit to top 3 per sentiment
                    review_text += f"Rating: {review['rating']}/5 - {review['text']}\n"
                
                chunks.append({
                    'text': review_text,
                    'type': f'reviews_{sentiment.lower()}'
                })
    
    return chunks

def split_text_smart(text: str, chunk_size: int = 200) -> List[str]:
    """Split text into chunks with smart sentence boundary detection"""
    if len(text) <= chunk_size:
        return [text]
    
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed chunk size, start new chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text using VADER"""
    scores = sentiment_analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_sentiment_color(sentiment: str) -> str:
    """Get color code for sentiment"""
    colors = {
        'Positive': '#28a745',
        'Negative': '#dc3545',
        'Neutral': '#6c757d'
    }
    return colors.get(sentiment, '#6c757d')

def format_price(price: float) -> str:
    """Format price with currency symbol"""
    return f"${price:,.2f}"

@st.cache_data
def load_product_data() -> List[Dict[str, Any]]:
    """Load all product data from JSON files"""
    products = []
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        return []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        products.extend(data)
                    else:
                        products.append(data)
            except Exception as e:
                st.error(f"Error loading {filename}: {str(e)}")
    
    # Ensure each product has required fields
    for product in products:
        if 'id' not in product:
            product['id'] = str(hash(product['title']))
        if 'rating' not in product:
            product['rating'] = 4.0
        if 'reviews' not in product:
            product['reviews'] = []
    
    return products

def generate_sample_data():
    """Generate sample product data for demonstration"""
    return [
        {
            "id": "laptop_001",
            "title": "TechPro UltraBook X1",
            "category": "Laptops",
            "price": 1299.99,
            "rating": 4.5,
            "description": "High-performance laptop with 16GB RAM, 512GB SSD, and Intel i7 processor. Perfect for professionals and content creators who need reliable performance for demanding tasks. Features a stunning 15.6-inch 4K display with excellent color accuracy.",
            "specs": {
                "Processor": "Intel Core i7-12700H",
                "RAM": "16GB DDR4",
                "Storage": "512GB NVMe SSD",
                "Display": "15.6-inch 4K IPS",
                "Graphics": "Intel Iris Xe",
                "Battery": "8+ hours",
                "Weight": "3.2 lbs"
            },
            "reviews": [
                {"rating": 5, "text": "Excellent laptop! Fast, reliable, and great build quality. Perfect for my work as a developer."},
                {"rating": 4, "text": "Good performance but battery life could be better for the price point."},
                {"rating": 5, "text": "Amazing display quality and very fast SSD. Highly recommend for creative work."}
            ]
        },
        {
            "id": "headphones_001",
            "title": "SoundWave Pro Wireless",
            "category": "Audio",
            "price": 249.99,
            "rating": 4.3,
            "description": "Premium wireless headphones with active noise cancellation and 30-hour battery life. Delivers studio-quality sound with deep bass and crystal-clear highs. Comfortable for all-day wear with memory foam padding.",
            "specs": {
                "Driver Size": "40mm",
                "Frequency Response": "20Hz - 20kHz",
                "Battery Life": "30 hours",
                "Charging": "USB-C fast charging",
                "Connectivity": "Bluetooth 5.2",
                "Weight": "280g",
                "Features": "ANC, Touch Controls"
            },
            "reviews": [
                {"rating": 5, "text": "Best headphones I've ever owned! Noise cancellation is incredible and sound quality is pristine."},
                {"rating": 4, "text": "Great sound quality but touch controls can be finicky sometimes."},
                {"rating": 4, "text": "Comfortable for long listening sessions, good value for money."}
            ]
        },
        {
            "id": "smartphone_001",
            "title": "NovaTech Galaxy S Pro",
            "category": "Smartphones",
            "price": 899.99,
            "rating": 4.7,
            "description": "Flagship smartphone with triple camera system, 5G connectivity, and all-day battery life. Features a gorgeous 6.7-inch OLED display with 120Hz refresh rate. Built with premium materials and IP68 water resistance.",
            "specs": {
                "Display": "6.7-inch OLED 120Hz",
                "Processor": "Snapdragon 8 Gen 2",
                "RAM": "12GB",
                "Storage": "256GB",
                "Camera": "108MP Triple Camera",
                "Battery": "5000mAh",
                "OS": "Android 14"
            },
            "reviews": [
                {"rating": 5, "text": "Fantastic phone! Camera quality is outstanding and performance is buttery smooth."},
                {"rating": 5, "text": "Love the display and battery life. Easily lasts a full day of heavy usage."},
                {"rating": 4, "text": "Great phone overall, though it's a bit pricey. Camera is definitely the highlight."}
            ]
        },
        {
            "id": "tablet_001",
            "title": "CreativePad Pro 12",
            "category": "Tablets",
            "price": 799.99,
            "rating": 4.4,
            "description": "Professional tablet designed for artists and creators. Comes with pressure-sensitive stylus and supports 4K video editing. Large 12.9-inch display with P3 wide color gamut for accurate color reproduction.",
            "specs": {
                "Display": "12.9-inch Liquid Retina",
                "Processor": "M2 Chip",
                "RAM": "8GB",
                "Storage": "128GB",
                "Stylus": "Included (4096 pressure levels)",
                "Battery": "10 hours",
                "Weight": "1.5 lbs"
            },
            "reviews": [
                {"rating": 5, "text": "Perfect for digital art! The stylus is incredibly responsive and accurate."},
                {"rating": 4, "text": "Great tablet but wish it came with more storage options at this price."},
                {"rating": 4, "text": "Excellent for note-taking and creative work. Display is beautiful."}
            ]
        },
        {
            "id": "smartwatch_001",
            "title": "FitTrack Elite",
            "category": "Wearables",
            "price": 349.99,
            "rating": 4.2,
            "description": "Advanced fitness smartwatch with comprehensive health monitoring, GPS tracking, and 7-day battery life. Water-resistant design perfect for swimmers and outdoor enthusiasts. Includes sleep tracking and stress monitoring.",
            "specs": {
                "Display": "1.4-inch AMOLED",
                "Battery Life": "7 days",
                "Water Resistance": "50m",
                "Sensors": "Heart Rate, SpO2, GPS",
                "Connectivity": "Bluetooth 5.0, WiFi",
                "Compatibility": "iOS/Android",
                "Weight": "45g"
            },
            "reviews": [
                {"rating": 4, "text": "Great fitness tracking features and battery life is impressive. Sleep tracking is very accurate."},
                {"rating": 4, "text": "Good smartwatch but app ecosystem could be better. Hardware is solid though."},
                {"rating": 5, "text": "Love the long battery life and accurate GPS. Perfect for my marathon training."}
            ]
        }
    ]