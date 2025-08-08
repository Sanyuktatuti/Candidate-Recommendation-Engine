# Candidate Recommendation Engine

An AI-powered web application that matches the best candidates to job descriptions using semantic similarity and machine learning.

## **Try It Now - Live Demo**

**Access the app instantly:** https://candidate-recommendation-engine-ks3gdxgzcjfto3624t55v2.streamlit.app/

### **How to Use:**

1. **Choose AI Service:**

   - **OpenAI (Recommended)**: Enter your OpenAI API key for best quality
   - **Free Mode**: No setup needed - works immediately

2. **Enter Job Details:**

   - Job title and detailed job description
   - Requirements and preferred skills

3. **Upload Resumes:**

   - Upload multiple PDF, DOCX, or TXT files
   - Or paste resume text directly

4. **Get Results:**

   - View ranked candidates with similarity scores
   - Read AI-generated fit summaries
   - Analyze charts and detailed breakdowns

5. **Pro Tip:** For professional recruiting, use OpenAI mode (~$0.002/candidate) for significantly better semantic understanding and summaries.

---

> **RECOMMENDATION**: For professional use, we highly recommend using **OpenAI API** for superior quality results. The app also includes free alternatives for testing and budget-conscious scenarios.

## Features

- **Smart Matching**: Uses AI embeddings and cosine similarity for semantic job-candidate matching
- **OpenAI Integration**: Recommended - Industry-leading semantic understanding and professional summaries
- **Free Fallback Options**: Works without API key using TF-IDF and keyword matching
- **Multiple Input Methods**: Upload resume files (PDF/DOCX/TXT) or paste text directly
- **AI-Powered Insights**: Generated summaries explaining why each candidate is a great fit
- **Interactive UI**: Modern Streamlit interface with charts, metrics, and detailed candidate analysis
- **Production Ready**: FastAPI backend with Docker deployment, health checks, and monitoring
- **Scalable Architecture**: Modular design with FAISS vector search and async processing

## Architecture

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

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- **Optional**: OpenAI API key (for best quality) or use free mode

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

Edit `.env` and add your OpenAI API key (recommended):

```env
OPENAI_API_KEY=your_openai_api_key_here  # Highly recommended for best quality
DEBUG=True
```

> **Pro Tip**: Get your OpenAI API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). The cost is minimal (~$2 for 100 candidate analyses) but the quality improvement is significant!

## AI Service Options

### **OpenAI Mode (Highly Recommended)**

- **Cost**: ~$0.002 per candidate analysis (~$2 for 100 candidates)
- **Technology**: OpenAI embeddings + GPT summaries
- **Quality**: Excellent semantic understanding
- **Speed**: Moderate (API calls)
- **Requirements**: OpenAI API key
- **Best For**: Professional HR use, production environments, high-quality results

> **Why We Recommend OpenAI:**
>
> - **Superior Accuracy**: Industry-leading semantic understanding
> - **Professional Summaries**: Human-like analysis of candidate fit
> - **Proven in Production**: Trusted by Fortune 500 companies
> - **Cost-Effective**: Excellent ROI for hiring decisions
> - **Continuous Improvement**: Benefits from OpenAI's latest models

### **Free Mode (Good Alternative)**

- **Cost**: $0 forever
- **Technology**: TF-IDF vectorization + keyword matching
- **Quality**: Good for basic screening
- **Speed**: Very fast
- **Requirements**: None
- **Best For**: Testing, demos, budget-conscious scenarios

### **Advanced Free (Optional)**

- **Cost**: $0 (but requires ~500MB download)
- **Technology**: SentenceTransformers + local models
- **Quality**: Very good semantic understanding
- **Speed**: Fast after initial setup
- **Requirements**: `pip install sentence-transformers`
- **Best For**: Users wanting quality without API costs

## Recommendation: Use OpenAI for Best Results

**For professional HR and recruiting use, we strongly recommend using OpenAI:**

| OpenAI Mode                     | Free Mode                |
| ------------------------------- | ------------------------ |
| Superior semantic understanding | Basic keyword matching   |
| Human-like candidate summaries  | Template-based summaries |
| Production-ready accuracy       | Good for testing/demos   |
| ~$2 per 100 candidates          | $0 cost                  |

**Bottom Line**: The quality difference is significant, and the cost is minimal for professional use.

### 3. Run the Application

**Option A: One-Command Launch (Recommended)**

```bash
# Start both servers automatically
./run.sh
```

**Note**: The local development uses `streamlit_app_local.py` (requires FastAPI backend), while the cloud deployment uses `streamlit_app.py` (standalone version).

**🛑 Stop All Services**

```bash
./stop.sh
```

** Check Status**

```bash
./status.sh
```

**Option B: Development Script**

```bash
python scripts/dev.py
```

**Option C: Manual Setup**

```bash
# For local development (with FastAPI backend)
pip install -r requirements_local.txt

# For cloud deployment (Streamlit only)
pip install -r requirements.txt

# Terminal 1: Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit (local development)
streamlit run streamlit_app_local.py --server.address 0.0.0.0 --server.port 8501
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

## Management Scripts

The application includes convenient management scripts for easy operation:

### `./run.sh` - Launch Application

- Automatically activates virtual environment
- Kills any processes using required ports (8000, 8501)
- Starts both FastAPI and Streamlit servers
- Performs health checks
- Runs in background with process management
- Creates log files (`api.log`, `streamlit.log`)

### `./stop.sh` - Stop All Services

- Gracefully stops all running servers
- Cleans up process files
- Kills any remaining processes on required ports

### `./status.sh` - Check Service Status

- Shows running processes and ports
- Performs health checks on both servers
- Displays recent log entries
- Shows access URLs and available commands

### Process Management

```bash
# Start everything
./run.sh

# Check if running
./status.sh

# Stop everything
./stop.sh

# View logs
tail -f api.log
tail -f streamlit.log
```

## Usage Guide

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

## Configuration

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

## How It Works

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

## Development

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
├── run.sh                   #  One-command launcher
├── stop.sh                  # 🛑 Stop all services
├── status.sh                #  Check service status
├── test_setup.py            # Setup validation
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

## Troubleshooting

### Common Issues

**API Not Starting:**

```bash
# Check service status
./status.sh

# Check if port is in use
lsof -i :8000

# Stop all and restart
./stop.sh
./run.sh

# Check logs
tail -f api.log
tail -f streamlit.log
```

**OpenAI API Errors:**

- Verify API key is correctly set in `.env`
- Check quota and billing in OpenAI dashboard
- Ensure proper network connectivity

**File Upload Issues:**

- Check file size (max 10MB by default)
- Verify file format (PDF/DOCX/TXT only)
- Ensure proper encoding for text files
- File upload section only appears after entering job title and description

**UI Issues:**

- If Streamlit shows errors, refresh the browser page
- Use incognito/private browsing to avoid cache issues
- Check browser console for JavaScript errors

**Performance Issues:**

- Reduce batch size for large candidate sets
- Consider caching for repeated queries
- Monitor API rate limits

## Performance Metrics

**OpenAI Mode Performance** ( recommended):

- **Total Processing Time**: 8.82 seconds for 3 candidates (2.94s per candidate)
- **Embedding Generation**: ~100ms per candidate
- **Similarity Search**: <10ms for 1000 candidates (FAISS)
- **AI Summary Generation**: ~500ms per candidate (professional quality)
- **File Processing**: ~200ms per PDF page
- **Quality**: Professional-grade semantic understanding

**Free Mode Performance** (alternative):

- **Total Processing Time**: ~1.5 seconds for 3 candidates (0.5s per candidate)
- **Embedding Generation**: ~50ms per candidate (TF-IDF)
- **Similarity Search**: <5ms for 100 candidates
- **Summary Generation**: ~10ms per candidate (template-based)
- **Quality**: Good for basic screening
- **Document Upload**: Supports multiple PDF/DOCX/TXT files simultaneously
- **Similarity Scores**: Range from 0-100% with detailed explanations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

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
