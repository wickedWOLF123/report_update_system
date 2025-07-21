# Development Setup Guide - Automated Report Updating System

## Prerequisites

- Python 3.9+ installed
- Git installed
- Node.js (for LangGraph Studio) - Download from [nodejs.org](https://nodejs.org/)

## Step-by-Step Setup

### 1. Clone and Navigate to Project
```bash
git clone <your-repo-url>
cd data-enrichment
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv ### 6. Install LangGraph CLI (for development)


# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install main dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Or install everything at once
pip install -e ".[dev]"
```

### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
# Required for MVP:
# - ANTHROPIC_API_KEY or OPENAI_API_KEY
# - TAVILY_API_KEY (for web search)
```

### 5. Install LangGraph Studio

#### Option A: Direct Download (Recommended)
1. Go to [LangGraph Studio Downloads](https://github.com/langchain-ai/langgraph-studio)
2. Download the installer for your OS
3. Install and launch

#### Option B: Via npm (if you have Node.js)
```bash
npm install -g @langchain/langgraph-studio
```

### 6. Install LangGraph CLI (for development)
```bash
# This should be installed with dev dependencies, but if not:
pip install -U "langgraph-cli[inmem]"
```

### 7. Verify Installation
```bash
# Test basic imports
python -c "import langgraph; import langchain; print('✅ LangGraph and LangChain installed successfully')"

# Test document processing imports
python -c "import fitz; import pdfplumber; print('✅ PDF processing libraries installed')"

# Test vector database imports  
python -c "import chromadb; print('✅ ChromaDB installed successfully')"

# Check LangGraph CLI
langgraph --help
```

## Running the Project

### 1. Start with LangGraph Studio (Recommended for Development)
```bash
# Open LangGraph Studio and point it to this directory
# Or if installed via CLI:
langgraph dev
```

### 2. Using Jupyter Notebook for Testing
```bash
# Start Jupyter
jupyter lab

# Open the testing notebook
# Navigate to ntbk/testing.ipynb
```

### 3. Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/enrichment_agent

# Run only unit tests
pytest tests/unit_tests/

# Run only integration tests
pytest tests/integration_tests/
```

### 4. Code Quality Tools
```bash
# Format code
black src/

# Lint code
ruff src/

# Type checking
mypy src/
```

## LangGraph Studio Usage

### Opening the Project
1. Launch LangGraph Studio
2. Open the project folder (`data-enrichment`)
3. Studio will automatically detect the `langgraph.json` configuration

### Testing in Studio
1. **Input Configuration**: Set your topic and extraction_schema
2. **Run Graph**: Execute the agent step by step
3. **Debug**: Inspect state at each node
4. **Modify**: Edit code and see changes via hot reload

### Studio Features for Our Project
- **Interactive Debugging**: Step through document analysis
- **State Inspection**: View extracted content and references
- **Tool Testing**: Test individual tools (PDF parsing, web search)
- **Graph Visualization**: See the workflow execution
- **Hot Reload**: Code changes apply immediately

## Development Workflow

### Daily Development Process
```bash
# 1. Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Pull latest changes
git pull origin main

# 3. Start LangGraph Studio
langgraph studio

# 4. Code, test, iterate

# 5. Run tests before committing
pytest

# 6. Commit changes
git add .
git commit -m "feat: implement document parsing tool"
git push origin main
```

### Testing Strategy
1. **Unit Tests**: Test individual functions and tools
2. **Integration Tests**: Test complete graph execution
3. **Studio Testing**: Interactive debugging and validation
4. **Manual Testing**: Real documents and edge cases

## Key Configuration Files

### `.env` File Contents
```bash
# LLM Provider (choose one)
ANTHROPIC_API_KEY=your_anthropic_key_here
# OR
OPENAI_API_KEY=your_openai_key_here

# Search API
TAVILY_API_KEY=your_tavily_key_here

# Optional: Database connections (for later)
# POSTGRES_URL=postgresql://user:pass@localhost/dbname
# REDIS_URL=redis://localhost:6379
```

### `langgraph.json` Configuration
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/enrichment_agent/graph.py:graph"
  },
  "env": ".env"
}
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get import errors, reinstall in development mode
pip uninstall enrichment-agent
pip install -e ".[dev]"
```

#### 2. PDF Processing Issues
```bash
# If PDF parsing fails, try installing system dependencies
# Ubuntu/Debian:
sudo apt-get install poppler-utils

# macOS:
brew install poppler

# Windows: Download poppler binaries and add to PATH
```

#### 3. ChromaDB Issues
```bash
# If ChromaDB has issues, try installing with conda
conda install -c conda-forge chromadb
```

#### 4. LangGraph Studio Not Loading
- Ensure Node.js is installed
- Try restarting Studio
- Check console for error messages
- Verify `.env` file has correct API keys

### Performance Optimization

#### For Large Documents
```python
# In configuration.py, adjust these settings:
max_document_size: int = 10_000_000  # Increase if needed
chunk_size: int = 1000  # Adjust for memory usage
```

#### For Vector Database
```python
# Use faster similarity search
# In tools.py, configure ChromaDB with:
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))
```

## Next Steps

1. **Test the Setup**: Try running the existing enrichment agent
2. **Create Sample Documents**: Prepare test PDFs for development
3. **Begin Phase 1**: Start implementing document analysis features
4. **Iterate**: Use Studio for testing and development

## Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangGraph Studio Guide](https://github.com/langchain-ai/langgraph-studio)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)

---

For any issues, check the troubleshooting section above or refer to the PROJECT_NOTES.md for comprehensive project details. 