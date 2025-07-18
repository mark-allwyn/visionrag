# Vision-RAG Multi-Model Chat Application

A comprehensive multi-modal RAG (Retrieval-Augmented Generation) system that supports multiple AI models and can process both text and images from documents.

## Features

- **Multi-Model Support**: Configurable AI models via JSON configuration file
- **Vision Capabilities**: Analyze images from PDFs and standalone image files
- **Document Processing**: Support for PDF, Word, PowerPoint, and image files
- **Environment Configuration**: Secure API key management via .env files
- **Persistent Sessions**: Maintain conversation history and vector store across sessions
- **Easy Model Management**: Add/remove models without code changes
- **Smart Image Processing**: Extract and analyze images from PDFs with context
- **Comprehensive Error Handling**: Robust fallback mechanisms and user feedback

## Model Configuration

The application uses a flexible JSON configuration system for managing AI models. See [MODEL_CONFIG.md](MODEL_CONFIG.md) for detailed instructions on adding and configuring models.

### Currently Supported Providers

- **OpenAI**: GPT-4 Vision, GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo
- **Google Gemini**: Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 1.5 Pro  
- **Anthropic Claude**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku

### Quick Model Update

To add a new model:
1. Edit `models_config.json`
2. Add your model configuration
3. Run `python validate_config.py` to verify
4. Restart the application

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required for embeddings (always needed)
COHERE_API_KEY=your_cohere_api_key_here

# Choose one or more AI providers
GENAI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Validate Configuration (Optional)

```bash
python validate_config.py
```

### 4. Run the Application

```bash
streamlit run visionrag.py
```

## Usage

1. **Select Model**: Use the dropdown in the sidebar to choose your preferred AI model
2. **Upload Documents**: Upload PDF, Word, PowerPoint, or image files
3. **Ask Questions**: Type your questions in the text input field and click "Ask"
4. **View Results**: Get comprehensive answers with source citations
5. **Clear Data**: Use "Clear All" to reset the knowledge base and start fresh

## Model Selection Guide

- **Vision Tasks**: Use vision-capable models for PDFs with images or standalone images
- **Text Only**: Any model works for text-only documents  
- **Performance**: Gemini models offer good performance and cost-effectiveness
- **Quality**: Claude models excel at detailed analysis and reasoning
- **Speed**: Gemini Flash is optimized for fast responses

## API Key Requirements

- **Cohere**: Required for all embeddings (text and vision) - **Always needed**
- **Google Gemini**: Required for Gemini models
- **OpenAI**: Required for GPT models  
- **Anthropic**: Required for Claude models

## File Support

- **PDF**: Text extraction + image analysis with vision models
- **Word (.docx)**: Text extraction with full document structure
- **PowerPoint (.pptx)**: Text and slide content extraction
- **Images**: PNG, JPG, JPEG, TIFF with vision analysis and context understanding

## Project Structure

```
visionrag/
├── visionrag.py           # Main Streamlit application
├── models_config.json     # AI model configurations
├── validate_config.py     # Configuration validator
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
├── original_notebook.ipynb # Original development notebook (reference)
├── uploaded_images/       # Runtime image storage (auto-created)
└── .env                  # Your API keys (create from .env.example)
```

## How Vision-RAG Works

Vision-RAG combines traditional text-based Retrieval-Augmented Generation with multimodal vision capabilities to process and understand both text and images from documents.

### Architecture Overview

The system uses a **dual-pathway approach** to handle both text and visual content:

#### 1. **Document Processing Pipeline**
- **Text Extraction**: Uses multiple fallback methods (PyMuPDF → pdfminer → UnstructuredLoader) for robust PDF text extraction
- **Image Extraction**: Extracts embedded images from PDFs with contextual information (page numbers, surrounding text)
- **Vision Analysis**: Processes images using Cohere's Embed v4.0 model for vision-based embeddings

#### 2. **Vector Storage System**
- **Vector Database**: **FAISS (Facebook AI Similarity Search)** for efficient similarity search
- **Custom Vector Store**: `VisionRAGVectorStore` class handles both text and image embeddings
- **Unified Embedding Strategy**:
  - **Single Model**: Cohere's `embed-v4.0` model for all content (text and images)
  - **Consistency**: Matches the original notebook approach exactly
  - **Simplicity**: One model for all embedding tasks ensures consistency
- **Hybrid Search**: Combines text and image similarity scores for comprehensive retrieval

#### 3. **Retrieval Process**
1. **Query Analysis**: Determines if the query requires text-only or multimodal search
2. **Parallel Search**: Simultaneously searches both text and image vector spaces
3. **Context Assembly**: Combines relevant text snippets and images with metadata
4. **Smart Ranking**: Returns top-k results (default: 6) with diverse content coverage

#### 4. **Generation Pipeline**
- **Multimodal Prompts**: Constructs prompts containing both text context and images
- **Model Selection**: Routes to appropriate AI model based on vision requirements
- **Response Generation**: AI models process combined text+image context to generate comprehensive answers

### Technical Stack

- **Vector Database**: FAISS-CPU for local, efficient similarity search
- **Embeddings**: Cohere Embed v4.0 (vision) + multilingual-22-12 (text)
- **Document Processing**: PyMuPDF, pdfminer, Unstructured, python-docx, python-pptx
- **AI Models**: OpenAI GPT, Google Gemini, Anthropic Claude (configurable)
- **Framework**: Streamlit for the user interface

### Key Innovations

1. **Context-Aware Image Processing**: Images are processed with surrounding text for better understanding
2. **Fallback Mechanisms**: Multiple extraction methods ensure robust document processing
3. **Dynamic Model Routing**: Automatically selects vision-capable models when images are present
4. **Persistent Storage**: Temporary image storage with automatic cleanup for security

### Performance Characteristics

- **Scalability**: FAISS enables efficient search across large document collections
- **Speed**: Local vector storage eliminates external database dependencies
- **Accuracy**: Dual embedding approach captures both semantic text meaning and visual content
- **Flexibility**: JSON-configurable models allow easy adaptation to new AI services

## Troubleshooting

- **Missing API Keys**: Check your `.env` file and ensure keys are properly set
- **Model Not Available**: Verify the corresponding API key is configured
- **Configuration Errors**: Run `python validate_config.py` to check your setup
- **Vision Errors**: Ensure you're using a vision-capable model for image analysis
- **Upload Issues**: Check file formats and ensure files aren't corrupted

## Security Notes

- Keep your `.env` file secure and never commit it to version control
- API keys are loaded from environment variables for security
- Images are temporarily stored and cleaned up automatically
- Processing logs can be cleared using the "Clear All" function

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python validate_config.py`
5. Submit a pull request

## License

This project is open source. See the license file for details.
