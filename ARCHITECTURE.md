# Architecture Documentation

## Modular Structure

The application has been refactored from a monolithic 2,000-line file into a clean, maintainable modular architecture suitable for enterprise development.

## Directory Structure

```
src/
├── __init__.py                 # Package initialization
├── models/                     # Data models and schemas
│   ├── __init__.py
│   └── candidate.py           # Candidate, JobDescription, SearchResult models
├── services/                   # Business logic and external integrations
│   ├── __init__.py
│   ├── document_processor.py  # Document processing service
│   ├── embedding_service.py   # Unified embedding service (OpenAI → Cohere → HF → TF-IDF)
│   └── ai_service.py          # Unified AI service (OpenAI → Enhanced Analysis)
├── utils/                      # Utility functions and helpers
│   ├── __init__.py
│   ├── similarity.py          # Similarity calculation utilities
│   └── result_processor.py    # Result processing and ranking
└── ui/                         # User interface components
    ├── __init__.py
    ├── styles.py              # CSS styles and themes
    ├── components.py          # Reusable UI components
    └── display.py             # Result display components
```

## Key Components

### Models (`src/models/`)

**Data Models with Validation:**
- `Candidate`: Represents a job candidate with validation
- `JobDescription`: Represents a job with validation and utility methods
- `SearchResult`: Contains search results with analytics properties

**Features:**
- Dataclass-based models with automatic validation
- Property methods for computed values (rankings, metrics)
- Type hints throughout for better IDE support

### Services (`src/services/`)

**Document Processor:**
- Handles PDF, DOCX, TXT file processing
- Graceful error handling with user feedback
- Support for multiple encodings

**Unified Embedding Service:**
- Automatic hierarchy: OpenAI → Cohere → Hugging Face → TF-IDF
- Intelligent fallback with service testing
- Advanced text preprocessing for better matching

**Unified AI Service:**
- OpenAI → Enhanced Analysis fallback
- Comprehensive candidate analysis with domain detection
- Professional summary generation

### Utils (`src/utils/`)

**Similarity Calculator:**
- Cosine similarity computation with scikit-learn
- Score normalization and adjustment for different services
- Similarity metrics calculation

**Result Processor:**
- Orchestrates the entire candidate processing pipeline
- Handles embeddings, similarities, and AI summaries
- Returns structured SearchResult objects

### UI (`src/ui/`)

**Styles:**
- Centralized CSS management
- Professional color scheme and typography
- Responsive design elements

**Components:**
- Reusable UI components (header, forms, inputs)
- Consistent user experience patterns
- Clean separation of UI logic

**Display:**
- Comprehensive result visualization
- Interactive charts with Plotly
- Tabbed interface for different views

## Benefits of Modular Architecture

### 🧪 **Testability**
- Each component can be unit tested independently
- Mock services for testing without API calls
- Clear interfaces make testing straightforward

### 🔧 **Maintainability**
- Single responsibility principle throughout
- Easy to locate and fix issues
- Clear dependencies between components

### 🚀 **Scalability**
- Easy to add new embedding services
- Simple to extend AI analysis capabilities
- UI components can be reused across views

### 👥 **Team Development**
- Multiple developers can work on different modules
- Clear ownership of components
- Reduced merge conflicts

### 📦 **Deployment**
- Still works on Streamlit Cloud with single entry point
- No changes needed to deployment process
- All components bundled together

## Import Strategy

The main application (`streamlit_app.py`) uses:

```python
# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Clean imports from modular components
from src.models.candidate import Candidate, JobDescription
from src.services.embedding_service import UnifiedEmbeddingService
from src.ui.components import UIComponents
```

This ensures:
- ✅ Works on Streamlit Cloud
- ✅ Clean import structure
- ✅ No external dependencies on file system layout
- ✅ Easy local development

## Application Flow

1. **Initialization** (`CandidateRecommendationApp.__init__`)
   - Initialize UI components
   - Create service instances
   - Set up result processor

2. **User Interface** (`CandidateRecommendationApp.run`)
   - Render header and sidebar
   - Handle job description input
   - Process candidate uploads/input

3. **Processing** (`ResultProcessor.process_candidates`)
   - Generate embeddings with service hierarchy
   - Calculate similarities
   - Generate AI summaries
   - Return structured results

4. **Display** (`ResultDisplay.display_search_results`)
   - Show detailed candidate results
   - Display similarity charts
   - Provide analytics insights

## Error Handling

- **Service Level**: Each service handles its own errors gracefully
- **UI Level**: User-friendly error messages with actionable guidance
- **Validation**: Data models validate input at creation time
- **Fallbacks**: Multiple service tiers ensure system always works

## Performance Considerations

- **Lazy Loading**: Services only initialize when needed
- **Batch Processing**: Efficient embedding generation
- **Caching**: UI components cache results in session state
- **Rate Limiting**: Built-in delays for free API tiers

This architecture provides a professional, enterprise-ready foundation while maintaining the simplicity of Streamlit Cloud deployment.
