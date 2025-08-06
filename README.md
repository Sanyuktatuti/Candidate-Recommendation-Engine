# 🎯 Candidate Recommendation Engine

An AI-powered web application that matches the best candidates to job descriptions using semantic similarity and machine learning.

## ✨ Features

- **Smart Matching**: Uses OpenAI embeddings and cosine similarity for semantic job-candidate matching
- **Multiple Input Methods**: Upload resume files (PDF/DOCX/TXT) or paste text directly
- **AI-Powered Insights**: GPT-generated summaries explaining why each candidate is a great fit
- **Interactive UI**: Modern Streamlit interface with charts, metrics, and detailed candidate analysis
- **Production Ready**: FastAPI backend with Docker deployment, health checks, and monitoring
- **Scalable Architecture**: Modular design with FAISS vector search and async processing

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │     OpenAI      │
│   Frontend      │────│    Backend      │────│      API        │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              │
                       ┌─────────────────┐
                       │     FAISS       │
                       │  Vector Index   │
                       │                 │
                       └─────────────────┘
```

### Tech Stack

- **Frontend**: Streamlit with Plotly for interactive charts
- **Backend**: FastAPI with async/await support
- **ML/AI**: OpenAI (embeddings + GPT), FAISS (vector similarity)
- **Document Processing**: PyPDF2, python-docx
- **Deployment**: Docker, Docker Compose, Nginx
- **Data**: In-memory vector store (easily extensible to PostgreSQL + pgvector)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- OpenAI API key

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/Candidate-Recommendation-Engine.git
cd Candidate-Recommendation-Engine
```

### 2. Environment Configuration

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
DEBUG=True
```

### 3. Quick Development Setup

**Option A: Development Script (Recommended)**
```bash
python scripts/dev.py
```

**Option B: Manual Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

### 4. Docker Deployment

```bash
# One-command deployment
./scripts/deploy.sh

# Or manually
docker-compose up -d
```

## 📍 Access Points

- **Streamlit App**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 💻 Usage Guide

### Using the Web Interface

1. **Job Description**: Enter the job title, description, and requirements
2. **Input Method**: Choose between:
   - **File Upload**: Upload PDF/DOCX/TXT resume files
   - **Text Input**: Manually enter candidate information
3. **Analysis**: Click "Analyze Candidates" to get:
   - Similarity scores (0-100%)
   - Ranked candidate list
   - AI-generated fit explanations
   - Interactive charts and metrics

### API Usage

**Search with JSON data:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": {
      "title": "Senior Python Developer",
      "description": "We need an experienced Python developer...",
      "requirements": "5+ years Python, FastAPI, ML experience"
    },
    "candidates": [
      {
        "name": "John Doe",
        "resume_text": "Experienced Python developer with 6 years..."
      }
    ],
    "top_k": 10,
    "include_summary": true
  }'
```

**Upload files:**
```bash
curl -X POST "http://localhost:8000/upload-search" \
  -F "job_title=Senior Python Developer" \
  -F "job_description=We need an experienced developer..." \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx"
```

## 🔧 Configuration

Key settings in `config.py`:

```python
# OpenAI Models
EMBEDDING_MODEL = "text-embedding-ada-002"  # For embeddings
CHAT_MODEL = "gpt-3.5-turbo"               # For summaries

# Limits
MAX_CANDIDATES = 50                         # Max candidates per request
MAX_FILE_SIZE = 10485760                   # 10MB file limit
VECTOR_DIMENSION = 1536                    # OpenAI embedding dimension

# Performance
RATE_LIMIT_REQUESTS = 100                  # Requests per hour
MAX_SUMMARY_LENGTH = 200                   # AI summary max length
```

## 📊 How It Works

### 1. Text Processing
- Extracts text from uploaded documents (PDF/DOCX/TXT)
- Cleans and preprocesses content
- Handles various file encodings and formats

### 2. Embedding Generation
- Uses OpenAI's `text-embedding-ada-002` model
- Generates 1536-dimensional vectors for job descriptions and resumes
- Batch processing for efficiency

### 3. Similarity Computation
- FAISS (Facebook AI Similarity Search) for fast vector operations
- Cosine similarity for semantic matching
- Normalized scores (0-100%)

### 4. AI Analysis
- GPT-powered explanations for each match
- Contextual summaries highlighting relevant skills
- Professional tone suitable for hiring managers

## 🛠️ Development

### Project Structure

```
candidate-recommendation-engine/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic data models
│   └── services/
│       ├── embedding_service.py    # OpenAI embeddings
│       ├── vector_service.py       # FAISS similarity search
│       ├── ai_service.py           # GPT summaries
│       └── document_service.py     # File processing
├── scripts/
│   ├── dev.py               # Development server
│   └── deploy.sh            # Deployment script
├── streamlit_app.py         # Frontend application
├── config.py                # Configuration
├── requirements.txt         # Dependencies
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service setup
└── README.md               # This file
```

### Adding New Features

**New Document Type:**
1. Add parser in `document_service.py`
2. Update `allowed_extensions` in config
3. Add validation in API endpoints

**Different Embedding Model:**
1. Update `EMBEDDING_MODEL` in config
2. Adjust `VECTOR_DIMENSION` if needed
3. Test compatibility with FAISS index

**Database Storage:**
1. Add SQLAlchemy models
2. Replace in-memory storage in `vector_service.py`
3. Add database migrations

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## 🚢 Production Deployment

### Environment Variables

```env
# Required
OPENAI_API_KEY=your_api_key

# Optional (with defaults)
DEBUG=False
API_HOST=0.0.0.0
API_PORT=8000
MAX_CANDIDATES=100
RATE_LIMIT_REQUESTS=1000
```

### Scaling Considerations

1. **Database**: Replace in-memory storage with PostgreSQL + pgvector
2. **Caching**: Add Redis for embedding/summary caching
3. **Background Jobs**: Use Celery for async processing
4. **Load Balancing**: Multiple API instances behind nginx
5. **Monitoring**: Prometheus + Grafana for metrics

### Security

- API key management via environment variables
- Input validation and sanitization
- File size and type restrictions
- Rate limiting on API endpoints
- CORS configuration for frontend access

## 🔍 Troubleshooting

### Common Issues

**API Not Starting:**
```bash
# Check if port is in use
lsof -i :8000

# Check logs
docker-compose logs api
```

**OpenAI API Errors:**
- Verify API key is correctly set in `.env`
- Check quota and billing in OpenAI dashboard
- Ensure proper network connectivity

**File Upload Issues:**
- Check file size (max 10MB by default)
- Verify file format (PDF/DOCX/TXT only)
- Ensure proper encoding for text files

**Performance Issues:**
- Reduce batch size for large candidate sets
- Consider caching for repeated queries
- Monitor API rate limits

## 📈 Performance Metrics

- **Embedding Generation**: ~100ms per candidate
- **Similarity Search**: <10ms for 1000 candidates (FAISS)
- **AI Summary**: ~500ms per candidate
- **File Processing**: ~200ms per PDF page

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for powerful embedding and language models
- **Meta AI** for FAISS vector similarity search
- **Streamlit** for rapid frontend development
- **FastAPI** for modern Python web framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Candidate-Recommendation-Engine/issues)
- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Email**: your.email@example.com

Built with ❤️ for better hiring decisions.