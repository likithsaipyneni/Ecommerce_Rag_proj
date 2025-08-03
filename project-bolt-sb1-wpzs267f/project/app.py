import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import json
import os

from rag import RAGSystem
from utils import load_product_data, get_sentiment_color, format_price

# Configure Streamlit page
st.set_page_config(
    page_title="AI Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .relevance-score {
        background-color: #007bff;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    return RAGSystem()

@st.cache_data
def load_data():
    """Load product data with caching"""
    return load_product_data()

def render_product_card(product: Dict[str, Any], relevance_score: float = None, show_reviews: bool = True):
    """Render a product card with styling"""
    with st.container():
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(product['title'])
            if relevance_score:
                st.markdown(f'<span class="relevance-score">Relevance: {relevance_score:.2f}</span>', 
                           unsafe_allow_html=True)
            
            st.write(f"**Category:** {product['category']}")
            st.write(f"**Price:** {format_price(product['price'])}")
            st.write(f"**Rating:** ⭐ {product['rating']}/5.0")
            
            # Description
            with st.expander("📝 Description"):
                st.write(product['description'])
            
            # Specifications
            if 'specs' in product and product['specs']:
                with st.expander("🔧 Specifications"):
                    for key, value in product['specs'].items():
                        st.write(f"**{key}:** {value}")
        
        with col2:
            # Price and action buttons
            st.metric("Price", format_price(product['price']))
            st.button(f"🛒 Add to Cart", key=f"cart_{product['id']}")
            st.button(f"❤️ Wishlist", key=f"wish_{product['id']}")
        
        # Reviews section
        if show_reviews and 'reviews' in product and product['reviews']:
            st.markdown("### 📝 Customer Reviews")
            for i, review in enumerate(product['reviews'][:3]):  # Show top 3 reviews
                sentiment_color = get_sentiment_color(review.get('sentiment', 'neutral'))
                st.markdown(f"""
                <div style="border-left: 3px solid {sentiment_color}; padding-left: 1rem; margin: 0.5rem 0;">
                    <strong>⭐ {review['rating']}/5</strong> - 
                    <span class="sentiment-{review.get('sentiment', 'neutral').lower()}">{review.get('sentiment', 'Neutral').title()}</span><br>
                    <em>"{review['text']}"</em>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.title("🛍️ AI-Powered Product Recommender")
    st.markdown("Find the perfect products using advanced AI recommendations")
    
    # Initialize systems
    rag_system = initialize_rag_system()
    products = load_data()
    
    # Sidebar for filters and preferences
    with st.sidebar:
        st.header("🎯 Preferences & Filters")
        
        # Category filter
        categories = list(set([p['category'] for p in products]))
        selected_categories = st.multiselect(
            "Categories", 
            categories, 
            default=categories
        )
        
        # Price range
        prices = [p['price'] for p in products]
        price_range = st.slider(
            "Price Range ($)",
            min_value=min(prices),
            max_value=max(prices),
            value=(min(prices), max(prices))
        )
        
        # Rating filter
        min_rating = st.slider(
            "Minimum Rating",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )
        
        # User preferences
        st.subheader("🎪 User Preferences")
        user_preferences = st.text_area(
            "Describe what you're looking for:",
            placeholder="E.g., budget-friendly laptop for gaming, eco-friendly products, high-performance electronics..."
        )
        
        # Apply filters
        filtered_products = [
            p for p in products 
            if p['category'] in selected_categories 
            and price_range[0] <= p['price'] <= price_range[1]
            and p['rating'] >= min_rating
        ]
        
        st.write(f"**{len(filtered_products)}** products match your filters")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search & Recommend", "📊 Compare Products", "📈 Analytics", "🗂️ Browse All"])
    
    with tab1:
        st.header("Search & AI Recommendations")
        
        # Search query
        search_query = st.text_input(
            "🔍 What are you looking for?",
            placeholder="E.g., wireless headphones with noise cancellation, gaming laptop under $1000..."
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_button = st.button("🚀 Search & Recommend", type="primary")
        
        with col2:
            max_results = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
        
        if search_button and search_query:
            with st.spinner("🤖 AI is analyzing products and generating recommendations..."):
                # Get recommendations from RAG system
                recommendations = rag_system.get_recommendations(
                    query=search_query,
                    products=filtered_products,
                    user_preferences=user_preferences,
                    max_results=max_results
                )
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommended products!")
                    
                    # AI-generated explanation
                    explanation = rag_system.generate_explanation(
                        query=search_query,
                        recommendations=recommendations[:3],
                        user_preferences=user_preferences
                    )
                    
                    if explanation:
                        st.markdown("### 🤖 AI Recommendation Explanation")
                        st.info(explanation)
                    
                    # Display recommendations
                    st.markdown("### 🎯 Recommended Products")
                    
                    for i, (product, score) in enumerate(recommendations):
                        st.markdown(f"#### #{i+1} Recommendation")
                        render_product_card(product, relevance_score=score)
                        st.markdown("---")
                
                else:
                    st.warning("No products found matching your criteria. Try adjusting your filters or search query.")
    
    with tab2:
        st.header("📊 Product Comparison")
        
        if len(filtered_products) >= 2:
            # Product selection for comparison
            product_options = {f"{p['title']} - {format_price(p['price'])}": p for p in filtered_products}
            
            col1, col2 = st.columns(2)
            
            with col1:
                product1_key = st.selectbox("Select first product:", list(product_options.keys()))
                product1 = product_options[product1_key]
            
            with col2:
                product2_key = st.selectbox("Select second product:", list(product_options.keys()))
                product2 = product_options[product2_key]
            
            if product1['id'] != product2['id']:
                st.markdown("### 🔄 Product Comparison")
                
                # Comparison table
                comparison_data = {
                    "Feature": ["Title", "Category", "Price", "Rating", "Description"],
                    product1['title'][:20] + "...": [
                        product1['title'],
                        product1['category'],
                        format_price(product1['price']),
                        f"⭐ {product1['rating']}/5",
                        product1['description'][:100] + "..."
                    ],
                    product2['title'][:20] + "...": [
                        product2['title'],
                        product2['category'],
                        format_price(product2['price']),
                        f"⭐ {product2['rating']}/5",
                        product2['description'][:100] + "..."
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Detailed comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### {product1['title']}")
                    render_product_card(product1, show_reviews=True)
                
                with col2:
                    st.markdown(f"#### {product2['title']}")
                    render_product_card(product2, show_reviews=True)
                
                # AI comparison
                if st.button("🤖 Get AI Comparison Analysis"):
                    with st.spinner("Analyzing products..."):
                        comparison_analysis = rag_system.compare_products(product1, product2)
                        if comparison_analysis:
                            st.markdown("### 🧠 AI Analysis")
                            st.info(comparison_analysis)
            else:
                st.warning("Please select two different products to compare.")
        else:
            st.warning("Need at least 2 products to enable comparison. Please adjust your filters.")
    
    with tab3:
        st.header("📈 Product Analytics")
        
        if filtered_products:
            # Price distribution
            fig_price = px.histogram(
                pd.DataFrame(filtered_products),
                x='price',
                title='Price Distribution',
                nbins=20
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Category distribution
            category_counts = pd.DataFrame(filtered_products)['category'].value_counts()
            fig_category = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='Products by Category'
            )
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Rating vs Price scatter
            df = pd.DataFrame(filtered_products)
            fig_scatter = px.scatter(
                df,
                x='price',
                y='rating',
                color='category',
                size='price',
                hover_data=['title'],
                title='Rating vs Price Analysis'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Sentiment analysis of reviews
            if st.button("📊 Analyze Review Sentiments"):
                sentiment_data = rag_system.analyze_all_sentiments(filtered_products)
                if sentiment_data:
                    fig_sentiment = px.bar(
                        x=list(sentiment_data.keys()),
                        y=list(sentiment_data.values()),
                        title="Overall Review Sentiment Distribution",
                        color=list(sentiment_data.values()),
                        color_continuous_scale="RdYlGn"
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with tab4:
        st.header("🗂️ Browse All Products")
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by:",
            ["Rating (High to Low)", "Price (Low to High)", "Price (High to Low)", "Name (A-Z)"]
        )
        
        # Apply sorting
        if sort_by == "Rating (High to Low)":
            sorted_products = sorted(filtered_products, key=lambda x: x['rating'], reverse=True)
        elif sort_by == "Price (Low to High)":
            sorted_products = sorted(filtered_products, key=lambda x: x['price'])
        elif sort_by == "Price (High to Low)":
            sorted_products = sorted(filtered_products, key=lambda x: x['price'], reverse=True)
        else:  # Name A-Z
            sorted_products = sorted(filtered_products, key=lambda x: x['title'])
        
        # Display products in grid
        cols = st.columns(2)
        for i, product in enumerate(sorted_products):
            with cols[i % 2]:
                render_product_card(product, show_reviews=False)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🤖 Powered by AI • Built with Streamlit & RAG Technology
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()