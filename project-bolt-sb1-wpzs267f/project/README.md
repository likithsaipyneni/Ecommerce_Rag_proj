# 🛍️ AI-Powered E-commerce Product Recommendation System

A comprehensive **RAG (Retrieval-Augmented Generation)** powered product recommendation system built with **Streamlit**. This application uses advanced AI techniques to provide intelligent product recommendations, comparisons, and insights.

## 🚀 Features

### Core Functionality
- **Multi-source Product Data**: Integrates title, description, specifications, and customer reviews
- **Advanced RAG System**: Uses HuggingFace SentenceTransformers with ChromaDB vector database
- **AI-Powered Recommendations**: Leverages Mistral-7B through HuggingFace Inference API
- **Sentiment Analysis**: VADER sentiment analysis for customer reviews
- **Product Comparison**: Side-by-side comparison tool with AI analysis
- **User Preferences**: Personalized recommendations based on user preferences and filters

### Novel Features
- **Smart Chunking Strategy**: Intelligent text chunking for optimal retrieval
- **Relevance Scoring**: Displays relevance scores for each recommendation
- **Top Review Display**: Shows most relevant reviews per query
- **Sentiment Visualization**: Color-coded sentiment indicators
- **Analytics Dashboard**: Product analytics with interactive charts

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ecommerce-rag-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Set up HuggingFace API** (for enhanced AI responses):
   ```bash
   export HUGGINGFACE_API_KEY="your-api-key-here"
   ```
   Or create a `.streamlit/secrets.toml` file:
   ```toml
   HUGGINGFACE_API_KEY = "your-api-key-here"
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the app**: Open your browser to `http://localhost:8501`

### Alternative: One-line Setup
```bash
pip install -r requirements.txt && streamlit run app.py
```

## 🌐 Deployment Instructions

### Hugging Face Spaces (Recommended)

1. **Create a new Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as the SDK
   - Set visibility to "Public"

2. **Upload files**:
   - Upload all project files to your Space
   - Ensure `requirements.txt` is in the root directory

3. **Configure secrets** (optional):
   - In your Space settings, add `HUGGINGFACE_API_KEY` as a secret
   - This enables enhanced AI responses

4. **Deploy**: Your Space will automatically build and deploy

### Local Deployment

For local production deployment:

```bash
# Install production server
pip install gunicorn

# Run with gunicorn (alternative to streamlit run)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🏗️ Architecture Overview

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   RAG System    │────│   ChromaDB      │
│   (app.py)      │    │   (rag.py)      │    │   (Vector DB)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │                        │              ┌─────────────────┐
         │                        │──────────────│ SentenceTransf. │
         │                        │              │  (Embeddings)   │
         │               ┌──────────────────┐    └─────────────────┘
         │               │  HuggingFace API │
         │               │  (Mistral-7B)    │
         │               └──────────────────┘
         │
┌─────────────────┐    ┌──────────────────┐
│   Utils         │    │   Product Data   │
│   (utils.py)    │    │   (data/*.json)  │
└─────────────────┘    └──────────────────┘
```

### Data Flow

1. **Data Ingestion**: Product data loaded from JSON files
2. **Text Processing**: Smart chunking of product information
3. **Embedding Generation**: SentenceTransformers creates vector embeddings
4. **Vector Storage**: ChromaDB stores embeddings for fast retrieval
5. **Query Processing**: User queries converted to embeddings
6. **Similarity Search**: ChromaDB finds most relevant products
7. **AI Generation**: Mistral-7B generates explanations and comparisons
8. **Sentiment Analysis**: VADER analyzes review sentiments
9. **UI Rendering**: Streamlit displays results with rich formatting

### File Structure

```
├── app.py                 # Main Streamlit application
├── rag.py                 # RAG system implementation
├── utils.py               # Utility functions (chunking, sentiment)
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── data/                 # Product data directory
│   ├── electronics.json  # Electronics products
│   ├── accessories.json  # Accessories and tablets
│   └── fitness.json      # Fitness equipment
└── chroma_db/            # ChromaDB storage (auto-created)
```

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **sentence-transformers**: Text embeddings (`all-MiniLM-L6-v2`)
- **chromadb**: Vector database for similarity search
- **transformers**: HuggingFace model integration
- **vaderSentiment**: Sentiment analysis
- **plotly**: Interactive visualizations

### Key Components

1. **Smart Chunking** (`utils.py`):
   - Separates product data into meaningful chunks
   - Groups reviews by sentiment
   - Optimizes chunk size for better retrieval

2. **RAG System** (`rag.py`):
   - Manages embedding generation and storage
   - Implements similarity search
   - Handles AI generation with fallbacks

3. **Streamlit UI** (`app.py`):
   - Multi-tab interface for different use cases
   - Interactive filters and preferences
   - Real-time search and recommendations

## 🤖 AI Model Information

### Primary Models
- **Embeddings**: `all-MiniLM-L6-v2` (384-dimensional, multilingual)
- **Generation**: `mistralai/Mistral-7B-Instruct-v0.1` (via HuggingFace API)
- **Sentiment**: VADER Sentiment Analyzer

### Fallback System
- Application works without HuggingFace API key
- Rule-based explanations and comparisons
- Maintains full functionality on free tier

## 📊 Sample Data

The system includes sample product data across multiple categories:
- **Electronics**: Laptops, smartphones, cameras
- **Audio**: Headphones, speakers
- **Fitness**: Treadmills, dumbbells
- **Home Appliances**: Vacuums, coffee machines, air purifiers

Each product includes:
- Basic information (title, price, rating, category)
- Detailed descriptions and specifications
- Multiple customer reviews with sentiment analysis

## 🔍 Usage Examples

### Search Queries
- "wireless headphones with noise cancellation"
- "budget laptop for programming under $1000"
- "fitness equipment for home gym"
- "camera for wedding photography"

### User Preferences
- "I prefer eco-friendly products"
- "Looking for budget-friendly options"
- "Need professional-grade equipment"
- "Want products with excellent customer support"

## 🚫 Limitations & Assumptions

### Assumptions
- Product data is stored locally in JSON format
- Users have basic familiarity with product categories
- Internet connection available for AI features (optional)

### Limitations
- ChromaDB runs locally (not suitable for distributed deployment)
- HuggingFace API has rate limits on free tier
- Sample data is limited for demonstration purposes
- No real-time inventory or pricing updates

### Free Tier Constraints
- No GPU requirements
- Works without paid API keys
- Uses efficient local models
- Minimal computational requirements

## 🔄 Extending the System

### Adding New Products
1. Create JSON files in the `data/` directory
2. Follow the existing schema structure
3. Include reviews with rating and text
4. Restart application to reindex

### Custom Models
- Replace `all-MiniLM-L6-v2` with other SentenceTransformer models
- Modify the embedding dimension in ChromaDB configuration
- Update model parameters in `rag.py`

### Additional Features
- Real-time inventory integration
- User authentication and preferences storage
- Purchase history analysis
- Advanced filtering options

## 📈 Performance Notes

- **Embedding Generation**: ~1-2 seconds for full product indexing
- **Query Response**: <500ms for similarity search
- **AI Generation**: 2-5 seconds (depending on API availability)
- **Memory Usage**: ~200-500MB for typical dataset

## 🆘 Troubleshooting

### Common Issues

1. **ChromaDB Import Error**:
   ```bash
   pip install --upgrade chromadb
   ```

2. **Sentence Transformers Download**:
   - First run downloads model (~80MB)
   - Ensure stable internet connection

3. **Missing Data**:
   - Verify JSON files exist in `data/` directory
   - Check JSON syntax validity

4. **HuggingFace API Issues**:
   - Verify API key is set correctly
   - Check rate limits and quota
   - Application works without API key (fallback mode)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**Built with ❤️ using Streamlit, HuggingFace, and ChromaDB**