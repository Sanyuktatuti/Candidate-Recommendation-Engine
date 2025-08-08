# Candidate Recommendation Engine

An AI-powered web application that matches the best candidates to job descriptions using semantic similarity and machine learning.

## **Try It Now - Live Demo**

**Access the app instantly:** https://candidate-recommendation-engine-ks3gdxgzcjfto3624t55v2.streamlit.app/

### **How to Use:**

1. **Smart AI Selection:**

   - **Automatic Mode**: App automatically uses the best available AI service
   - **No Setup Needed**: Premium features work instantly with our API keys
   - **Intelligent Fallback**: Seamlessly switches between services for optimal performance

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

**AI Service Tiers (Automatic based on the availability of the models):**

- **🚀 Premium**: OpenAI API (Best quality)
- **✨ Professional**: Cohere API (Excellent quality)
- **⚡ Enhanced**: Hugging Face API (Good quality)
- **📊 Basic**: TF-IDF Analysis (Always available)

---

> **NEW**: **Automatic Smart Selection** - Our app automatically uses the best available AI service, providing premium quality with intelligent fallback to ensure continuous operation.

## 🏠 **Run Locally**

Want to run the app on your own machine? It's easy!

### **Quick Start (One Command):**
```bash
./run_local.sh
```

### **Manual Setup:**
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Run the app
streamlit run streamlit_app.py
```

### **🔑 Local Benefits:**
- ✅ **Uses Your API Keys**: Automatically reads from `.env` file
- ✅ **Same Premium Experience**: Identical to cloud version
- ✅ **Fast Development**: Test changes instantly
- ✅ **Offline Capable**: Works without internet (TF-IDF mode)

**📱 App opens at:** `http://localhost:8501`

---

## Features

- **Automatic AI Service Selection**: Intelligent hierarchy - Premium → Professional → Enhanced → Basic
- **Smart Matching**: Uses AI embeddings and cosine similarity for semantic job-candidate matching
- **Multi-Tier AI Integration**: OpenAI, Cohere, Hugging Face APIs with intelligent fallback
- **Zero Setup Required**: Premium features work instantly with pre-configured API access
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
- **ML/AI**:
  - **Premium**: OpenAI (embeddings + GPT)
  - **Professional**: Cohere (Embed v3 + Command R)
  - **Enhanced**: Hugging Face Inference API
  - **Basic**: Enhanced TF-IDF + Local models
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF2, python-docx
- **Deployment**: Docker, Docker Compose, Nginx, Streamlit Cloud
- **Data**: In-memory vector store (easily extensible to PostgreSQL + pgvector)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- **Optional**: Your own API keys (automatically uses our premium services if none provided)

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

Edit `.env` and add your API keys (optional - app works without them):

```env
OPENAI_API_KEY=your_openai_api_key_here    # Optional: For premium quality
COHERE_API_KEY=your_cohere_api_key_here    # Optional: For professional quality
HF_API_TOKEN=your_hf_token_here            # Optional: For enhanced quality
DEBUG=True
```

> **Note**: All API keys are optional! The app automatically uses our premium services if you don't provide your own keys. If you want to use your own APIs, get keys at:
>
> - OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
> - Cohere: [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)
> - Hugging Face: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Automatic AI Service Hierarchy

Our app automatically selects the best available AI service in this priority order:

### **🚀 Tier 1: Premium (OpenAI)**

- **Technology**: OpenAI text-embedding-ada-002 + GPT-3.5/4
- **Quality**: Industry-leading semantic understanding
- **Use Case**: Professional HR, production environments
- **Features**: Premium embeddings + human-like summaries
- **Availability**: Automatic (no setup required)

### **✨ Tier 2: Professional (Cohere)**

- **Technology**: Cohere Embed v3.0 + Command R
- **Quality**: Excellent semantic understanding with multilingual support
- **Use Case**: Professional recruiting, international hiring
- **Features**: High-quality embeddings + enhanced analysis
- **Availability**: Automatic fallback

### **⚡ Tier 3: Enhanced (Hugging Face)**

- **Technology**: BGE/E5/GTE models via Inference API
- **Quality**: Good semantic understanding with fast processing
- **Use Case**: Medium-scale recruiting, budget-conscious teams
- **Features**: Quality embeddings + sophisticated analysis
- **Availability**: Automatic fallback

### **📊 Tier 4: Basic (TF-IDF + Local Models)**

- **Technology**: Enhanced TF-IDF + SentenceTransformers (when available)
- **Quality**: Reliable keyword and phrase matching
- **Use Case**: Testing, demos, offline environments
- **Features**: Advanced preprocessing + template analysis
- **Availability**: Always available (guaranteed fallback)

## Smart Selection Benefits

✅ **Zero Configuration**: Works immediately with premium quality  
✅ **Intelligent Fallback**: Seamlessly switches if one service is unavailable  
✅ **Cost Effective**: Uses our API allocations efficiently  
✅ **Consistent Experience**: Users always get the best available quality  
✅ **Production Ready**: Built-in redundancy for enterprise use

### 3. Run the Application

**Option A: One-Command Launch (Recommended)**

```bash
# Start both servers automatically
./run.sh
```

**Note**: Both local and cloud versions now use the same automatic AI hierarchy for consistent premium experience.

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
3. **Automatic AI Selection**: App automatically uses best available service tier
4. **Analysis**: Click "Analyze Candidates" to get:
   - Similarity scores (0-100%)
   - Ranked candidate list
   - AI-generated fit explanations
   - Interactive charts and metrics
   - Service tier indicator showing which AI is active

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
# API Keys (all optional - app works without them)
OPENAI_API_KEY = ""                        # Premium tier
COHERE_API_KEY = ""                        # Professional tier
HF_API_TOKEN = ""                          # Enhanced tier

# AI Models (automatic selection)
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI embeddings
CHAT_MODEL = "gpt-3.5-turbo"               # OpenAI summaries
COHERE_MODEL = "embed-english-v3.0"        # Cohere embeddings
HF_MODEL = "BAAI/bge-large-en"             # Hugging Face embeddings

# Limits
MAX_CANDIDATES = 50                         # Max candidates per request
MAX_FILE_SIZE = 10485760                   # 10MB file limit
VECTOR_DIMENSION = 1536                    # Default dimension (varies by service)

# Performance
RATE_LIMIT_REQUESTS = 100                  # Requests per hour
MAX_SUMMARY_LENGTH = 200                   # AI summary max length
```

## How It Works

### 1. Text Processing

- Extracts text from uploaded documents (PDF/DOCX/TXT)
- Cleans and preprocesses content
- Handles various file encodings and formats

### 2. Automatic Service Selection & Embedding Generation

- **Tier 1**: OpenAI `text-embedding-ada-002` (1536-dim vectors)
- **Tier 2**: Cohere `embed-english-v3.0` (1024-dim vectors)
- **Tier 3**: Hugging Face BGE/E5 models (384-768-dim vectors)
- **Tier 4**: Enhanced TF-IDF with domain knowledge (2000-dim vectors)
- Intelligent fallback between services
- Batch processing for efficiency

### 3. Similarity Computation

- FAISS (Facebook AI Similarity Search) for fast vector operations
- Cosine similarity for semantic matching
- Normalized scores (0-100%) with tier-appropriate adjustments
- Consistent scoring across different embedding services

### 4. AI Analysis & Summaries

- **Premium**: OpenAI GPT-powered explanations with human-like analysis
- **Professional**: Enhanced template analysis with sophisticated matching
- **Enhanced**: Advanced preprocessing with domain detection
- **Basic**: Keyword and phrase analysis with professional formatting
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
│       ├── embedding_service.py    # Multi-tier embeddings (OpenAI, Cohere, HF, TF-IDF)
│       ├── vector_service.py       # FAISS similarity search
│       ├── ai_service.py           # Multi-tier AI summaries
│       └── document_service.py     # File processing
├── scripts/
│   ├── dev.py               # Development server
│   └── deploy.sh            # Deployment script
├── run.sh                   #  One-command launcher
├── stop.sh                  # 🛑 Stop all services
├── status.sh                #  Check service status
├── streamlit_app.py         # Cloud frontend (standalone with unified services)
├── streamlit_app_local.py   # Local frontend (FastAPI backend integration)
├── config.py                # Multi-tier configuration
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

### **🚀 Premium Tier (OpenAI) Performance**

- **Total Processing Time**: 8.82 seconds for 3 candidates (2.94s per candidate)
- **Embedding Generation**: ~100ms per candidate
- **Similarity Search**: <10ms for 1000 candidates (FAISS)
- **AI Summary Generation**: ~500ms per candidate (GPT-powered)
- **Quality**: Industry-leading semantic understanding
- **Best For**: Professional recruiting, high-stakes hiring

### **✨ Professional Tier (Cohere) Performance**

- **Total Processing Time**: ~6 seconds for 3 candidates (2.0s per candidate)
- **Embedding Generation**: ~80ms per candidate
- **Similarity Search**: <10ms for 1000 candidates
- **AI Summary Generation**: ~200ms per candidate (enhanced analysis)
- **Quality**: Excellent semantic understanding with multilingual support
- **Best For**: International recruiting, diverse candidate pools

### **⚡ Enhanced Tier (Hugging Face) Performance**

- **Total Processing Time**: ~4 seconds for 3 candidates (1.3s per candidate)
- **Embedding Generation**: ~60ms per candidate
- **Similarity Search**: <8ms for 1000 candidates
- **AI Summary Generation**: ~100ms per candidate (sophisticated analysis)
- **Quality**: Good semantic understanding with fast processing
- **Best For**: Medium-scale recruiting, quick screening

### **📊 Basic Tier (TF-IDF) Performance**

- **Total Processing Time**: ~1.5 seconds for 3 candidates (0.5s per candidate)
- **Embedding Generation**: ~50ms per candidate (enhanced TF-IDF)
- **Similarity Search**: <5ms for 100 candidates
- **Summary Generation**: ~10ms per candidate (template-based)
- **Quality**: Reliable keyword matching with domain knowledge
- **Best For**: Testing, demos, offline environments

### **Universal Features**

- **File Processing**: ~200ms per PDF page (all tiers)
- **Document Upload**: Supports multiple PDF/DOCX/TXT files simultaneously
- **Similarity Scores**: Range from 0-100% with tier-appropriate explanations
- **Automatic Fallback**: <1s switching time between tiers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for industry-leading embedding and language models
- **Cohere** for excellent multilingual semantic understanding
- **Hugging Face** for open-source transformer models and inference infrastructure
- **Meta AI** for FAISS vector similarity search
- **Streamlit** for rapid frontend development and cloud deployment
- **FastAPI** for modern Python web framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Candidate-Recommendation-Engine/issues)
- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Email**: your.email@example.com

Built for better hiring decisions.
