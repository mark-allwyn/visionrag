import os
import tempfile
from pathlib import Path
import base64
import io
import numpy as np
import time
import random
import traceback
import json

import streamlit as st
import openai
import anthropic
import google.generativeai as genai
import cohere
from dotenv import load_dotenv
import fitz


# Load environment variables
load_dotenv()
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from PIL import Image

# Check if PyMuPDF is available
try:
    # fitz is already imported at the top
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Check for PDF text extraction dependencies
try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

# Model Configuration
def load_model_config():
    """Load model configuration from JSON file"""
    config_path = Path(__file__).parent / "models_config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('models', {}), config.get('default_model', 'Gemini 2.5 Pro')
    except FileNotFoundError:
        st.error(f"Model configuration file not found: {config_path}")
        st.info("Please ensure models_config.json exists in the same directory as this script.")
        # Fallback to minimal config
        return {
            "Gemini 2.5 Pro": {
                "provider": "gemini",
                "model": "models/gemini-2.5-pro",
                "requires_key": "GENAI_API_KEY",
                "vision_capable": True
            }
        }, "Gemini 2.5 Pro"
    except json.JSONDecodeError as e:
        st.error(f"Error parsing model configuration file: {e}")
        st.info("Please check the JSON syntax in models_config.json.")
        return {}, "Gemini 2.5 Pro"
    except Exception as e:
        st.error(f"Error loading model configuration: {e}")
        return {}, "Gemini 2.5 Pro"

# Load model configuration from file
MODEL_CONFIG, DEFAULT_MODEL = load_model_config()

# API Keys from environment variables - remove hardcoded fallbacks for security
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize API clients
genai.configure(api_key=GENAI_API_KEY)
def initialize_clients():
    """Initialize API clients based on available keys"""
    clients = {}
    
    # Cohere (always needed for embeddings)
    if COHERE_API_KEY:
        try:
            clients['cohere'] = cohere.ClientV2(api_key=COHERE_API_KEY)
        except Exception as e:
            st.error(f"Failed to initialize Cohere client: {str(e)}")
            clients['cohere'] = None
    
    # Gemini
    if GENAI_API_KEY:
        try:
            # Store reference to genai module instead of creating a custom client
            clients['gemini'] = genai
        except Exception as e:
            st.error(f"Failed to configure Gemini client: {str(e)}")
            clients['gemini'] = None
    
    # OpenAI
    if OPENAI_API_KEY:
        try:
            import openai
            clients['openai'] = openai.OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            st.warning("OpenAI library not installed. Run: pip install openai")
            clients['openai'] = None
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
            clients['openai'] = None
    
    # Claude/Anthropic
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            clients['claude'] = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except ImportError:
            st.warning("Anthropic library not installed. Run: pip install anthropic")
            clients['claude'] = None
        except Exception as e:
            st.error(f"Failed to initialize Claude client: {str(e)}")
            clients['claude'] = None
    
    return clients

# Initialize clients with error handling
@st.cache_resource
def get_api_clients():
    """Get API clients with caching for performance"""
    try:
        return initialize_clients()
    except Exception as e:
        st.error(f"Critical error initializing API clients: {str(e)}")
        return {}

# Get cached API clients
api_clients = get_api_clients()
co = api_clients.get('cohere')

# Embedding model - using embed-v4.0 for both text and vision (matching notebook approach)
EMBEDDING_MODEL = "embed-v4.0"

# Custom logging system
class ProcessLogger:
    def __init__(self):
        if 'process_logs' not in st.session_state:
            st.session_state.process_logs = []
    
    def log(self, message, level="info"):
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.process_logs.append(f"[{timestamp}] {message}")
        # Keep only last 50 messages to prevent memory issues
        if len(st.session_state.process_logs) > 50:
            st.session_state.process_logs = st.session_state.process_logs[-50:]
    
    def show_logs(self):
        if st.session_state.process_logs:
            with st.expander("Processing Logs", expanded=False):
                log_text = "\n".join(st.session_state.process_logs)
                st.text_area("", value=log_text, height=200, disabled=True, key="log_display")
    
    def clear(self):
        st.session_state.process_logs = []

# Initialize global logger
logger = ProcessLogger()

# Max resolution for images (same as notebook)
max_pixels = 1568*1568

def cleanup_old_images():
    """Clean up old uploaded images on app start"""
    upload_dir = Path("uploaded_images")
    if upload_dir.exists():
        for file in upload_dir.glob("*"):
            try:
                # Keep recently uploaded files (less than 1 hour old)
                if file.stat().st_mtime < time.time() - 3600:  # 1 hour
                    file.unlink()
            except:
                pass

# Clean up old images when the app starts
cleanup_old_images()

def clear_vector_store():
    """Clear all vector store data, images, memory, and uploaded files"""
    try:
        # Clear uploaded images directory
        upload_dir = Path("uploaded_images")
        if upload_dir.exists():
            for file in upload_dir.glob("*"):
                try:
                    file.unlink()
                except:
                    pass
        
        # Clear session state data
        if 'vector_store' in st.session_state:
            del st.session_state['vector_store']
        if 'history' in st.session_state:
            st.session_state.history = []
        
        # Clear processing notifications
        if 'show_processing_notification' in st.session_state:
            del st.session_state['show_processing_notification']
        if 'processing_summary' in st.session_state:
            del st.session_state['processing_summary']
        
        # Force file uploader to reset by incrementing its key
        if 'uploader_key' not in st.session_state:
            st.session_state.uploader_key = 0
        st.session_state.uploader_key += 1
        
        # Also clear chat input
        if 'chat_key' not in st.session_state:
            st.session_state.chat_key = 0
        st.session_state.chat_key += 1
        
        # Clear process logs
        logger.clear()
        
        st.success("🗑️ **Knowledge base cleared successfully!**\n\n" + 
                  "✓ Uploaded files removed from interface\n\n" +
                  "✓ Processed documents deleted\n\n" + 
                  "✓ Images and vector store cleared\n\n" +
                  "✓ Conversation history deleted\n\n" +
                  "✓ Processing notifications cleared\n\n" +
                  "✓ Processing logs cleared")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing knowledge base: {str(e)}")

def call_model_with_retry(prompt, model_name=None, max_retries=3, base_delay=1):
    """
    Call the selected AI model with retry logic for handling errors.
    """
    # Use selected model from session state or default from config
    if model_name is None:
        model_name = st.session_state.get('selected_model', DEFAULT_MODEL)
    
    if model_name not in MODEL_CONFIG:
        st.error(f"Unknown model: {model_name}")
        return "Error: Unknown model selected."
    
    model_config = MODEL_CONFIG[model_name]
    
    # Check if required API key is available
    if model_config["requires_key"]:
        key_name = model_config["requires_key"]
        if not os.getenv(key_name):
            st.error(f"API key {key_name} not found in environment variables. Please set it in .env file.")
            return f"Error: Missing API key for {model_name}."
    
    # Check if vision is required but model doesn't support it
    has_images = any(isinstance(item, Image.Image) for item in prompt if not isinstance(item, str))
    if has_images and not model_config["vision_capable"]:
        st.error(f"Model {model_name} does not support vision. Please select a vision-capable model.")
        return f"Error: {model_name} does not support image analysis."
    
    provider = model_config["provider"]
    model_id = model_config["model"]
    
    # Check if the required client is available
    if provider not in api_clients or api_clients[provider] is None:
        st.error(f"Client for {provider} not initialized. Please check your API key and restart the app.")
        return f"Error: {provider} client not available."
    
    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                return call_gemini_model(prompt, model_id, attempt, max_retries, base_delay)
            elif provider == "openai":
                return call_openai_model(prompt, model_id, attempt, max_retries, base_delay)
            elif provider == "claude":
                return call_claude_model(prompt, model_id, attempt, max_retries, base_delay)
            else:
                st.error(f"Unsupported provider: {provider}")
                return f"Error: Unsupported provider {provider}."
                
        except Exception as e:
            error_msg = str(e).lower()
            if "overloaded" in error_msg or "unavailable" in error_msg or "resource exhausted" in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    st.warning(f"Model {model_name} is overloaded. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    st.error(f"Model {model_name} is overloaded after {max_retries} attempts.")
                    return f"Error: {model_name} is currently overloaded. Please try again later."
            else:
                st.error(f"Error with model {model_name}: {str(e)}")
                return f"Error: {str(e)}"
    
    return "Error: Maximum retries exceeded."

def convert_prompt_for_gemini(prompt):
    """Convert prompt to the format expected by the Gemini SDK"""
    if isinstance(prompt, str):
        # Simple text prompt
        return prompt
    elif isinstance(prompt, list):
        # Multi-modal prompt (text + images)
        converted_content = []
        for item in prompt:
            if isinstance(item, str):
                converted_content.append(item)
            elif isinstance(item, Image.Image):
                # For PIL Images, pass them directly - the current google.generativeai handles this
                converted_content.append(item)
            elif isinstance(item, dict):
                # Already in the correct format for Gemini
                converted_content.append(item)
        return converted_content
    else:
        # For other types, return as is
        return prompt

def call_gemini_model(prompt, model_id, attempt, max_retries, base_delay):
    """Call Gemini model"""
    if 'gemini' not in api_clients or api_clients['gemini'] is None:
        raise Exception("Gemini client not initialized. Check GENAI_API_KEY.")
    
    try:
        # Convert prompt to the format expected by the new SDK
        converted_prompt = convert_prompt_for_gemini(prompt)
        
        # Create the GenerativeModel instance
        model = api_clients['gemini'].GenerativeModel(model_id)
        
        # Generate content with the new usage pattern
        response = model.generate_content(converted_prompt, stream=False)
        
        return response.text
    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")

def call_claude_model(prompt, model_id, attempt, max_retries, base_delay):
    """Call Claude model"""
    if 'claude' not in api_clients or api_clients['claude'] is None:
        raise Exception("Claude client not initialized. Check ANTHROPIC_API_KEY and install anthropic library.")
    
    try:
        # Convert prompt format for Claude
        messages = []
        
        # Handle different prompt formats
        if isinstance(prompt, list):
            # Multi-modal prompt (text + images)
            content = []
            for item in prompt:
                if isinstance(item, str):
                    content.append({"type": "text", "text": item})
                elif isinstance(item, Image.Image):
                    # Convert PIL Image to base64
                    buffered = io.BytesIO()
                    item.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_str
                        }
                    })
            
            messages.append({"role": "user", "content": content})
        else:
            # Text-only prompt
            messages.append({"role": "user", "content": prompt})
        
        response = api_clients['claude'].messages.create(
            model=model_id,
            max_tokens=1000,
            messages=messages
        )
        
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Claude API error: {str(e)}")

def call_openai_model(prompt, model_id, attempt, max_retries, base_delay):
    """Call OpenAI model"""
    if 'openai' not in api_clients or api_clients['openai'] is None:
        raise Exception("OpenAI client not initialized. Check OPENAI_API_KEY and install openai library.")
    
    try:
        # Convert prompt format for OpenAI
        messages = []
        current_message = {"role": "user", "content": []}
        
        for item in prompt:
            if isinstance(item, str):
                current_message["content"].append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                item.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                current_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                })
        
        messages.append(current_message)
        
        response = api_clients['openai'].chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

# Helper functions from the notebook
def resize_image(pil_image):
    """Resize too large images"""
    org_width, org_height = pil_image.size
    
    # Resize image if too large
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

def base64_from_image(img_path):
    """Convert images to a base64 string before sending it to the API"""
    pil_image = Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"
    
    resize_image(pil_image)
    
    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")
    
    return img_data

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using available libraries.
    Try PyMuPDF first, then pdfminer, then UnstructuredPDFLoader.
    """
    logger.log("Starting PDF text extraction...")
    text_content = ""
    
    # Method 1: Try PyMuPDF
    if PYMUPDF_AVAILABLE:
        try:
            logger.log("Attempting text extraction with PyMuPDF...")
            doc = fitz.open(pdf_path)
            logger.log(f"PDF has {len(doc)} pages")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.get_text()
            doc.close()
            logger.log(f"Successfully extracted text using PyMuPDF ({len(text_content)} characters)")
            return text_content
        except Exception as e:
            logger.log(f"PyMuPDF text extraction failed: {str(e)}")
    
    # Method 2: Try pdfminer
    if PDFMINER_AVAILABLE:
        try:
            logger.log("Attempting text extraction with pdfminer...")
            from pdfminer.high_level import extract_text
            text_content = extract_text(pdf_path)
            logger.log(f"Successfully extracted text using pdfminer ({len(text_content)} characters)")
            return text_content
        except Exception as e:
            logger.log(f"pdfminer text extraction failed: {str(e)}")
    
    # Method 3: Try UnstructuredPDFLoader
    try:
        logger.log("Attempting text extraction with UnstructuredPDFLoader...")
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        text_content = "\n\n".join([doc.page_content for doc in docs])
        logger.log(f"Successfully extracted text using UnstructuredPDFLoader ({len(text_content)} characters)")
        return text_content
    except Exception as e:
        logger.log(f"UnstructuredPDFLoader failed: {str(e)}")
        return None
def extract_images_from_pdf(pdf_path):
    """
    Extract images from PDF and process them with vision-based approach.
    """
    if not PYMUPDF_AVAILABLE:
        logger.log("PyMuPDF not available. PDF images will not be processed with vision analysis.")
        return []
    
    logger.log("Starting PDF image extraction and processing...")
    
    try:
        pdf_document = fitz.open(pdf_path)
        images = []
        total_images = 0
        
        # Count total images first
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            total_images += len(page.get_images(full=True))
        
        logger.log(f"Found {total_images} images across {len(pdf_document)} pages")
        
        processed_images = 0
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True)
            
            if image_list:
                logger.log(f"Processing page {page_num + 1}: {len(image_list)} images found")
            
            # Get page text for context
            page_text = page.get_text()
            page_words = page_text.split()
            
            for img_index, img in enumerate(image_list):
                processed_images += 1
                logger.log(f"Processing image {processed_images}/{total_images} (Page {page_num + 1}, Image {img_index + 1})")
                
                # Extract image data
                xref = img[0]
                pix = fitz.Pixmap(pdf_document, xref)
                
                # Convert to PIL Image
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("ppm")
                    pil_image = Image.open(io.BytesIO(img_data))
                    
                    logger.log(f"Image dimensions: {pil_image.size[0]}x{pil_image.size[1]} pixels")
                    
                    # Get image position and dimensions
                    img_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                    
                    # Extract surrounding text context
                    surrounding_text = ""
                    if img_rect:
                        logger.log("Extracting surrounding text context...")
                        # Get text blocks near the image
                        text_blocks = page.get_text("dict")["blocks"]
                        for block in text_blocks:
                            if "lines" in block:
                                block_rect = fitz.Rect(block["bbox"])
                                # Check if text block is near the image
                                if (abs(block_rect.y1 - img_rect.y0) < 100 or 
                                    abs(block_rect.y0 - img_rect.y1) < 100):
                                    for line in block["lines"]:
                                        for span in line["spans"]:
                                            surrounding_text += span["text"] + " "
                    
                    # Truncate surrounding text if too long
                    if len(surrounding_text) > 500:
                        surrounding_text = surrounding_text[:500] + "..."
                    
                    if surrounding_text:
                        logger.log(f"Found {len(surrounding_text)} characters of surrounding text")
                    
                    # Save image to persistent location
                    upload_dir = Path("uploaded_images")
                    upload_dir.mkdir(exist_ok=True)
                    persistent_img_path = upload_dir / f"pdf_img_p{page_num}_{img_index}_{hash(pdf_path)}.png"
                    pil_image.save(persistent_img_path)
                    logger.log(f"Saved image to: {persistent_img_path.name}")
                    
                    # Process with vision-based approach with enhanced metadata
                    logger.log("Generating vision embeddings with Cohere Embed v4...")
                    vision_docs = process_image_with_vision(
                        str(persistent_img_path), 
                        f"PDF_Page_{page_num+1}_Image_{img_index+1}",
                        additional_metadata={
                            "page_number": page_num + 1,
                            "image_index": img_index + 1,
                            "total_pages": len(pdf_document),
                            "surrounding_text": surrounding_text.strip(),
                            "page_text_length": len(page_words),
                            "image_position": {
                                "x": img_rect.x0 if img_rect else 0,
                                "y": img_rect.y0 if img_rect else 0,
                                "width": img_rect.width if img_rect else 0,
                                "height": img_rect.height if img_rect else 0
                            } if img_rect else None,
                            "context_type": "pdf_embedded_image"
                        }
                    )
                    images.extend(vision_docs)
                    logger.log(f"Successfully processed image {processed_images}/{total_images}")
                
                pix = None
        
        pdf_document.close()
        logger.log(f"Completed PDF image processing: {len(images)} images successfully processed")
        return images
        
    except Exception as e:
        logger.log(f"Could not extract images from PDF: {str(e)}")
        return []

def process_pdf_with_vision(pdf_path, filename):
    """
    Process PDF with both text extraction and vision-based image analysis.
    """
    docs = []
    
    # Extract text using fallback methods
    try:
        text_content = extract_text_from_pdf(pdf_path)
        if text_content:
            # Create a Document object for the text content
            text_doc = Document(
                page_content=text_content,
                metadata={
                    "source": filename,
                    "type": "text",
                    "extraction_method": "pdf_text"
                }
            )
            docs.append(text_doc)
            logger.log(f"Successfully extracted text from PDF: {filename}")
        else:
            logger.log(f"Could not extract text from PDF: {filename}")
    except Exception as e:
        logger.log(f"Error extracting text from PDF {filename}: {str(e)}")
    
    # Extract and process images with vision analysis
    try:
        image_docs = extract_images_from_pdf(pdf_path)
        if image_docs:
            docs.extend(image_docs)
            logger.log(f"Successfully extracted and processed {len(image_docs)} images from PDF: {filename}")
        else:
            logger.log(f"No images found in PDF: {filename}")
    except Exception as e:
        logger.log(f"Could not process PDF images from {filename}: {str(e)}")
    
    return docs

def process_image_with_vision(image_path, filename, additional_metadata=None):
    """
    Process image using Cohere Embed v4 for vision-based RAG.
    """
    try:
        if not co:
            logger.log(f"Cohere client not available. Skipping vision processing for {filename}")
            return []
        
        # Check if image file exists
        if not os.path.exists(image_path):
            logger.log(f"Image file not found: {image_path}")
            return []
            
        logger.log(f"Converting {filename} to base64 format...")
        
        # Get the base64 representation of the image with error handling
        try:
            image_base64 = base64_from_image(image_path)
        except Exception as e:
            logger.log(f"Error converting image to base64: {str(e)}")
            return []
            
        api_input_document = {
            "content": [
                {"type": "image", "image": image_base64},
            ]
        }

        logger.log("Calling Cohere Embed v4.0 for vision embedding...")
        # Call the Embed v4.0 model with the image information
        try:
            api_response = co.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                inputs=[api_input_document],
            )
        except Exception as e:
            logger.log(f"Error calling Cohere Embed API: {str(e)}")
            return []

        # Get the embedding
        try:
            emb = np.asarray(api_response.embeddings.float[0])
            logger.log(f"Generated embedding vector: {emb.shape[0]} dimensions")
        except Exception as e:
            logger.log(f"Error processing embedding response: {str(e)}")
            return []
        
        # Build enhanced page content with context
        page_content = f"Image: {filename}"
        if additional_metadata:
            if additional_metadata.get("surrounding_text"):
                page_content += f"\n\nSurrounding text context: {additional_metadata['surrounding_text']}"
            if additional_metadata.get("page_number"):
                page_content += f"\n\nLocation: Page {additional_metadata['page_number']}"
                if additional_metadata.get("total_pages"):
                    page_content += f" of {additional_metadata['total_pages']}"
        
        logger.log("Building document metadata...")
        # Build comprehensive metadata
        metadata = {
            "source": filename,
            "type": "image_vision",
            "embedding": emb.tolist(),
            "image_path": image_path
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Create a Document object for the image
        doc = Document(
            page_content=page_content,
            metadata=metadata
        )
        logger.log(f"Successfully created document object for {filename}")
        return [doc]
    except Exception as e:
        logger.log(f"Error processing image {filename}: {str(e)}")
        return []

def embed_documents(docs):
    """
    Generate embeddings for documents using Cohere.
    """
    if not docs:
        return []
    
    embedder = CohereEmbeddings(
        cohere_api_key=COHERE_API_KEY, 
        model=EMBEDDING_MODEL,
        user_agent="visionrag-streamlit-app"
    )
    return embedder.embed_documents([d.page_content for d in docs])

def build_vector_store(docs, embeddings):
    """
    Build a custom vector store that handles both text and vision embeddings.
    """
    if not docs:
        logger.log("No documents to process for vector store")
        return None
    
    logger.log(f"Building vector store with {len(docs)} documents...")
    
    # Separate text docs and image docs
    text_docs = [doc for doc in docs if doc.metadata.get("type") != "image_vision"]
    image_docs = [doc for doc in docs if doc.metadata.get("type") == "image_vision"]
    
    logger.log(f"Text documents: {len(text_docs)}")
    logger.log(f"Image documents: {len(image_docs)}")
    
    # Create a simple custom vector store
    class VisionRAGVectorStore:
        def __init__(self, text_docs, image_docs):
            logger.log("Initializing VisionRAG vector store...")
            self.text_docs = text_docs
            self.image_docs = image_docs
            
            # Create embeddings for text docs if any
            if text_docs:
                logger.log(f"Generating embeddings for {len(text_docs)} text documents...")
                try:
                    embedder = CohereEmbeddings(
                        cohere_api_key=COHERE_API_KEY, 
                        model=EMBEDDING_MODEL,
                        user_agent="visionrag-streamlit-app"
                    )
                    self.text_embeddings = embedder.embed_documents([d.page_content for d in text_docs])
                    logger.log(f"Generated {len(self.text_embeddings)} text embeddings")
                except Exception as e:
                    logger.log(f"Error generating text embeddings: {str(e)}")
                    self.text_embeddings = []
            else:
                self.text_embeddings = []
                
            # Get image embeddings from metadata
            self.image_embeddings = [doc.metadata["embedding"] for doc in image_docs]
            if image_docs:
                logger.log(f"Loaded {len(self.image_embeddings)} pre-computed image embeddings")
        
        def as_retriever(self):
            return VisionRAGRetriever(self)
    
    class VisionRAGRetriever:
        def __init__(self, vector_store):
            self.vector_store = vector_store
            
        def get_relevant_documents(self, query, k=6):  # Increased from 3 to 6 for better coverage
            # Search both text and image embeddings
            results = []
            
            # Search text documents
            if self.vector_store.text_docs:
                try:
                    # Get query embedding for text search
                    embedder = CohereEmbeddings(
                        cohere_api_key=COHERE_API_KEY, 
                        model=EMBEDDING_MODEL,
                        user_agent="visionrag-streamlit-app"
                    )
                    query_emb = embedder.embed_query(query)
                    
                    # Calculate similarities for text docs
                    if self.vector_store.text_embeddings:
                        text_similarities = np.dot(query_emb, np.array(self.vector_store.text_embeddings).T)
                        top_text_indices = np.argsort(text_similarities)[-k:][::-1]
                        text_results = [self.vector_store.text_docs[i] for i in top_text_indices]
                        results.extend(text_results)
                except Exception as e:
                    logger.log(f"Error in text search: {str(e)}")
            
            # Search image documents
            if self.vector_store.image_docs and co:
                try:
                    # Get query embedding for image search
                    api_response = co.embed(
                        model="embed-v4.0",
                        input_type="search_query",
                        embedding_types=["float"],
                        texts=[query],
                    )
                    query_emb = np.asarray(api_response.embeddings.float[0])
                    
                    # Calculate similarities for image docs
                    if self.vector_store.image_embeddings:
                        image_similarities = np.dot(query_emb, np.array(self.vector_store.image_embeddings).T)
                        top_image_indices = np.argsort(image_similarities)[-k:][::-1]
                        image_results = [self.vector_store.image_docs[i] for i in top_image_indices]
                        results.extend(image_results)
                except Exception as e:
                    logger.log(f"Error in image search: {str(e)}")
            
            # Return mixed results but prefer diverse content
            return results[:k*2]  # Return more results to ensure good coverage
    
    return VisionRAGVectorStore(text_docs, image_docs)

def load_and_process_files(uploaded_files):
    """
    Load and vectorize uploaded files (PDF, Word, PPT).
    """
    logger.log(f"Starting file processing for {len(uploaded_files)} files...")
    docs = []
    
    # Check if Cohere client is available for vision processing
    if not co:
        logger.log("ERROR: Cohere client not initialized. Cannot process files without embeddings.")
        st.error("Cohere API key is missing or invalid. Please check your .env file.")
        return None
    
    # Create a directory to store uploaded images
    upload_dir = Path("uploaded_images")
    upload_dir.mkdir(exist_ok=True)
    logger.log(f"Upload directory created/verified: {upload_dir}")
    
    for i, up in enumerate(uploaded_files):
        try:
            logger.log(f"Processing file {i+1}/{len(uploaded_files)}: {up.name}")
            suffix = Path(up.name).suffix.lower()
            logger.log(f"File type detected: {suffix}")
            
            if suffix in (".png", ".jpg", ".jpeg", ".tiff"):
                logger.log(f"Processing as image file: {up.name}")
                # For images, save to persistent location
                image_path = upload_dir / up.name
                
                try:
                    file_data = up.read()
                    logger.log(f"Read {len(file_data)} bytes from {up.name}")
                except Exception as read_error:
                    logger.log(f"Failed to read file {up.name}: {str(read_error)}")
                    continue
                
                # Validate file data
                if not file_data:
                    logger.log(f"Empty file data for {up.name}")
                    continue
                    
                try:
                    with open(image_path, "wb") as f:
                        f.write(file_data)
                    logger.log(f"Saved image to: {image_path}")
                except Exception as save_error:
                    logger.log(f"Failed to save image {up.name}: {str(save_error)}")
                    continue
                
                # Verify file was saved correctly
                if not image_path.exists():
                    logger.log(f"Failed to save image file: {image_path}")
                    continue
                
                logger.log(f"Processing image with vision capabilities...")
                # Process image with vision-based approach
                try:
                    loaded_docs = process_image_with_vision(
                        str(image_path), 
                        up.name,
                        additional_metadata={
                            "context_type": "standalone_image",
                            "file_size": len(file_data),
                            "upload_timestamp": time.time()
                        }
                    )
                    if loaded_docs:
                        logger.log(f"Successfully processed image: {up.name} -> {len(loaded_docs)} documents")
                        docs.extend(loaded_docs)
                    else:
                        logger.log(f"Vision processing returned no documents for: {up.name}")
                except Exception as vision_error:
                    logger.log(f"Vision processing failed for {up.name}: {str(vision_error)}")
                    continue
                    
            else:
                logger.log(f"Processing as document file: {up.name}")
                # For other file types, use temporary files
                try:
                    file_data = up.read()
                    logger.log(f"Read {len(file_data)} bytes from document {up.name}")
                except Exception as read_error:
                    logger.log(f"Failed to read document {up.name}: {str(read_error)}")
                    continue
                
                if not file_data:
                    logger.log(f"Empty document data for {up.name}")
                    continue
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(file_data)
                        tmp_file_path = tmp_file.name
                    logger.log(f"Created temporary file: {tmp_file_path}")
                except Exception as temp_error:
                    logger.log(f"Failed to create temporary file for {up.name}: {str(temp_error)}")
                    continue
                
                loaded_docs = None
                try:
                    if suffix == ".pdf":
                        logger.log("Using enhanced PDF processing with vision analysis...")
                        # Use enhanced PDF processing with vision analysis
                        loaded_docs = process_pdf_with_vision(tmp_file_path, up.name)
                    elif suffix in (".docx", ".doc"):
                        logger.log("Loading Word document...")
                        try:
                            loader = UnstructuredWordDocumentLoader(tmp_file_path)
                            loaded_docs = loader.load()
                            logger.log(f"Word loader returned {len(loaded_docs) if loaded_docs else 0} documents")
                        except Exception as word_error:
                            logger.log(f"Word document processing failed: {str(word_error)}")
                            loaded_docs = []
                    elif suffix in (".pptx", ".ppt"):
                        logger.log("Loading PowerPoint presentation...")
                        try:
                            loader = UnstructuredPowerPointLoader(tmp_file_path)
                            loaded_docs = loader.load()
                            logger.log(f"PowerPoint loader returned {len(loaded_docs) if loaded_docs else 0} documents")
                        except Exception as ppt_error:
                            logger.log(f"PowerPoint processing failed: {str(ppt_error)}")
                            loaded_docs = []
                    else:
                        logger.log(f"Unsupported file type: {suffix}")
                        loaded_docs = []
                except Exception as doc_error:
                    logger.log(f"Document processing error for {up.name}: {str(doc_error)}")
                    loaded_docs = []
                finally:
                    # Clean up temporary file for non-image files
                    try:
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                            logger.log(f"Cleaned up temporary file: {tmp_file_path}")
                    except Exception as cleanup_error:
                        logger.log(f"Failed to cleanup temp file: {str(cleanup_error)}")
                
                if loaded_docs and len(loaded_docs) > 0:
                    docs.extend(loaded_docs)
                    logger.log(f"Added {len(loaded_docs)} documents from {up.name}")
                else:
                    logger.log(f"No documents extracted from {up.name}")
            
        except Exception as e:
            logger.log(f"Critical error processing file {up.name}: {str(e)}")
            import traceback
            logger.log(f"Traceback: {traceback.format_exc()}")
            continue
    
    logger.log(f"Document processing complete. Total docs collected: {len(docs)}")
    
    if not docs:
        logger.log("ERROR: No documents were successfully loaded. Processing failed.")
        # Show more specific error message
        st.error("Failed to process any documents. Common issues:")
        st.error("• Missing Cohere API key for embeddings")
        st.error("• Unsupported file format or corrupted files")
        st.error("• Missing required libraries (check requirements.txt)")
        st.info("Check the processing logs below for detailed error information.")
        return None
    
    logger.log(f"Building vector store with {len(docs)} documents...")
    try:
        vs = build_vector_store(docs, None)
        if vs:
            logger.log("Vector store built successfully")
            return vs
        else:
            logger.log("ERROR: Vector store creation failed")
            st.error("Failed to create vector store from processed documents.")
            return None
    except Exception as vs_error:
        logger.log(f"Vector store creation error: {str(vs_error)}")
        import traceback
        logger.log(f"Traceback: {traceback.format_exc()}")
        st.error(f"Vector store creation failed: {str(vs_error)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Vision-RAG Chat", layout="wide")
st.title("Vision-RAG Conversational QA")

# Sidebar with organized controls
with st.sidebar:
    st.header("Vision-RAG Assistant")
    
    # Model Configuration Section
    st.markdown("### Model Configuration")
    
    # Filter available models based on API keys AND client initialization
    available_models = []
    unavailable_models = []
    
    for model_name, config in MODEL_CONFIG.items():
        provider = config["provider"]
        key_required = config["requires_key"]
        
        # Check if API key is available
        if key_required and not os.getenv(key_required):
            unavailable_models.append(model_name)
        # Check if client is properly initialized
        elif provider not in api_clients or api_clients[provider] is None:
            unavailable_models.append(model_name)
        else:
            available_models.append(model_name)
    
    if not available_models:
        st.error("No models available. Please check your API keys in .env file.")
        st.info("The app requires at least one AI model to function properly.")
        st.stop()  # Stop the app execution
    
    # Initialize selected model in session state
    if 'selected_model' not in st.session_state:
        # Use DEFAULT_MODEL if available, otherwise fall back to first available model
        if DEFAULT_MODEL in available_models:
            st.session_state.selected_model = DEFAULT_MODEL
        else:
            st.session_state.selected_model = available_models[0] if available_models else None
    
    # Ensure selected model is still available
    if st.session_state.selected_model not in available_models:
        # Try DEFAULT_MODEL first, then fall back to first available
        if DEFAULT_MODEL in available_models:
            st.session_state.selected_model = DEFAULT_MODEL
        else:
            st.session_state.selected_model = available_models[0]
    
    selected_model = st.selectbox(
        "Select AI Model:",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        help="Choose the AI model for processing your questions and documents"
    )
    
    # Update session state
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.rerun()
    
    # Model capabilities indicator
    if selected_model in MODEL_CONFIG:
        model_config = MODEL_CONFIG[selected_model]
        col1, col2 = st.columns(2)
        with col1:
            if model_config["vision_capable"]:
                st.success("Vision Capable")
            else:
                st.warning("Text Only")
        with col2:
            provider = model_config["provider"].title()
            st.info(f"Provider: {provider}")
        

    # Show unavailable models if any
    if unavailable_models:
        with st.expander("Configure Additional Models", expanded=False):
            st.markdown("**Missing API Keys or Failed Initialization:**")
            for model_name in unavailable_models:
                config = MODEL_CONFIG[model_name]
                key_needed = config["requires_key"]
                provider = config["provider"]
                
                if key_needed and not os.getenv(key_needed):
                    st.markdown(f"- **{model_name}**: Missing API key `{key_needed}`")
                elif provider not in api_clients or api_clients[provider] is None:
                    st.markdown(f"- **{model_name}**: Client initialization failed for {provider}")
                else:
                    st.markdown(f"- **{model_name}**: Unknown issue")
            st.markdown("Add missing keys to your `.env` file to enable more models.")
    
    # Debug: Show client status
    if st.checkbox("Show API Client Status", value=False):
        st.markdown("**API Client Status:**")
        for provider, client in api_clients.items():
            status = "✅ Initialized" if client is not None else "❌ Failed"
            st.markdown(f"- {provider.title()}: {status}")
    
    st.divider()
    
    # System Status Section
    st.markdown("### System Status")
    
    # Knowledge base status
    if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
        st.success("Knowledge Base Active")
        
        # Count documents if possible
        try:
            text_docs = len(st.session_state.vector_store.text_docs)
            image_docs = len(st.session_state.vector_store.image_docs)
            st.markdown(f"**Text documents:** {text_docs}")
            st.markdown(f"**Image documents:** {image_docs}")
        except:
            pass
    else:
        st.info("No Knowledge Base Loaded")
    
    # Conversation history status
    if st.session_state.get('history'):
        history_count = len(st.session_state.history)
        st.success(f"**Conversation:** {history_count} Q&A pairs")
    else:
        st.info("No Conversation History")
    
    st.divider()
    
    # Actions Section
    st.markdown("### Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear All", 
                    type="secondary", 
                    help="Delete all uploaded files, processed documents, vector store, and conversation history",
                    use_container_width=True):
            clear_vector_store()
    
    with col2:
        if st.button("Show Logs", 
                    type="secondary", 
                    help="View processing logs",
                    use_container_width=True):
            logger.show_logs()
    
    st.divider()
    
    # Help Section
    st.markdown("### Quick Help")
    with st.expander("Supported File Types", expanded=False):
        st.markdown("""
        **Documents:**
        - PDF (text + images)
        - Word (.docx, .doc)
        - PowerPoint (.pptx, .ppt)
        
        **Images:**
        - PNG, JPG, JPEG, TIFF
        """)
    
    with st.expander("Model Capabilities", expanded=False):
        st.markdown("""
        **Vision Models:**
        - Analyze images and documents
        - Extract text from images
        - Understand charts and diagrams
        
        **Text-Only Models:**
        - Process text documents
        - Answer questions about content
        - Faster processing
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Vision-RAG v2.0*", help="Enhanced RAG system with vision capabilities")

# File uploader
st.markdown("### Upload Documents")

# Initialize uploader key if not exists
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_files = st.file_uploader(
    "Choose files to analyze", 
    type=["pdf","docx","pptx","png","jpg","jpeg","tiff"],
    accept_multiple_files=True,
    help="Upload PDF, Word, PowerPoint documents or image files",
    key=f"file_uploader_{st.session_state.uploader_key}"
)

# Chat Interface
# Show processing notifications if files have been processed
if st.session_state.get('show_processing_notification', False):
    processing_summary = st.session_state.get('processing_summary', [])
    if processing_summary:
        st.success("**Documents processed successfully!**")
        for summary in processing_summary:
            st.markdown(f"<small>{summary}</small>", unsafe_allow_html=True)
        st.info("**Ready for questions!** You can now ask questions about your documents using the chat interface below.")
    else:
        st.success("Documents processed successfully!")
        st.info("💡 **Ready for questions!** You can now ask questions about your documents using the chat interface below.")

# Process uploaded files if any
if uploaded_files and 'vector_store' not in st.session_state:
    logger.clear()
    with st.spinner("Processing your documents..."):
        vector_store = load_and_process_files(uploaded_files)
        if vector_store:
            st.session_state.vector_store = vector_store
            
            # Store processing summary for notification display
            processing_summary = []
            for log_entry in st.session_state.process_logs:
                if "Successfully extracted text from PDF:" in log_entry:
                    filename = log_entry.split("Successfully extracted text from PDF: ")[1]
                    processing_summary.append(f"✓ Text extracted from {filename}")
                elif "Successfully extracted and processed" in log_entry and "images from PDF:" in log_entry:
                    parts = log_entry.split("Successfully extracted and processed ")[1].split(" images from PDF: ")
                    count = parts[0]
                    filename = parts[1]
                    processing_summary.append(f"✓ {count} images processed from {filename}")
                elif "Successfully processed image:" in log_entry:
                    filename = log_entry.split("Successfully processed image: ")[1]
                    processing_summary.append(f"✓ Image processed: {filename}")
            
            # Store summary and flag for notification display
            st.session_state.processing_summary = processing_summary
            st.session_state.show_processing_notification = True
            st.rerun()  # Rerun to show notifications
        else:
            st.error("Failed to process documents. Please check your files and try again.")
    
    # Show processing logs on main page
    logger.show_logs()

st.markdown("### Ask Questions")
if uploaded_files or ('vector_store' in st.session_state and st.session_state.vector_store is not None):
    # Initialize history if not exists
    if "history" not in st.session_state:
        st.session_state.history = []

    # Chat input with submit button
    col1, col2 = st.columns([4, 1])
    
    # Initialize chat key if not exists
    if 'chat_key' not in st.session_state:
        st.session_state.chat_key = 0
    
    with col1:
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know about your documents?",
            help="Ask questions about the content in your uploaded files",
            key=f"chat_input_{st.session_state.chat_key}",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_query = st.button("Ask", type="primary", use_container_width=True)
    
    # Process query only when submit button is clicked and query is provided
    if submit_query and query:
        try:
            # Use existing vector store or the newly created one
            vector_store = st.session_state.get('vector_store') or (load_and_process_files(uploaded_files) if uploaded_files else None)
            
            if vector_store:
                st.session_state.vector_store = vector_store
                
                with st.spinner("Analyzing your question..."):
                    # Retrieve relevant docs
                    retriever = vector_store.as_retriever()
                    docs = retriever.get_relevant_documents(query)

                    # Check if we have image documents
                    image_docs = [d for d in docs if d.metadata.get("type") == "image_vision"]
                    text_docs = [d for d in docs if d.metadata.get("type") != "image_vision"]
                    
                    if image_docs:
                        # Vision + text based answering
                        text_context = ""
                        if text_docs:
                            text_context = "\n\n".join([d.page_content for d in text_docs])
                        
                        # Process images
                        image_contexts = []
                        valid_image_paths = []
                        
                        for i, image_doc in enumerate(image_docs):
                            image_path = image_doc.metadata["image_path"]
                            
                            if Path(image_path).exists():
                                valid_image_paths.append(image_path)
                                
                                image_context = f"Image {i+1}: {image_doc.metadata['source']}"
                                if image_doc.metadata.get("page_number"):
                                    image_context += f" (Page {image_doc.metadata['page_number']}"
                                    if image_doc.metadata.get("total_pages"):
                                        image_context += f" of {image_doc.metadata['total_pages']}"
                                    image_context += ")"
                                
                                if image_doc.metadata.get("surrounding_text"):
                                    image_context += f"\nContext: {image_doc.metadata['surrounding_text']}"
                                
                                image_contexts.append(image_context)
                        
                        if valid_image_paths:
                            # Create prompt with images and text
                            prompt_text = f"""Answer the question based on the following images and text content. 
Analyze ALL the images and text provided to give a comprehensive answer.

Text Content:
{text_context if text_context else "No additional text content available."}

Image Information:
{chr(10).join(image_contexts)}

Question: {query}"""
                            
                            try:
                                prompt = [prompt_text]
                                for image_path in valid_image_paths:
                                    prompt.append(Image.open(image_path))
                                
                                selected_model = st.session_state.get('selected_model', DEFAULT_MODEL)
                                answer = call_model_with_retry(prompt)
                                
                                # Display answer
                                st.markdown(f"**Answer:** {answer}")
                                
                                # Show sources in expandable sections
                                if valid_image_paths:
                                    with st.expander(f"Source Images ({len(valid_image_paths)})", expanded=False):
                                        for i, (image_path, image_doc) in enumerate(zip(valid_image_paths, image_docs)):
                                            st.image(image_path, caption=f"Image {i+1}: {image_doc.metadata['source']}", width=300)
                                            
                                            # Show page information
                                            if image_doc.metadata.get("page_number"):
                                                st.caption(f"Page {image_doc.metadata['page_number']}" + 
                                                         (f" of {image_doc.metadata['total_pages']}" if image_doc.metadata.get("total_pages") else ""))
                                            
                                            # Show surrounding text context if available
                                            if image_doc.metadata.get("surrounding_text"):
                                                surrounding_text = image_doc.metadata["surrounding_text"].strip()
                                                if surrounding_text:
                                                    st.markdown("**Surrounding text context:**")
                                                    st.markdown(f'"{surrounding_text}"')
                                            
                                            # Show image position if available
                                            if image_doc.metadata.get("image_position"):
                                                pos = image_doc.metadata["image_position"]
                                                if pos and pos.get("x") is not None:
                                                    st.caption(f"Position: x={pos['x']:.1f}, y={pos['y']:.1f}, width={pos['width']:.1f}, height={pos['height']:.1f}")
                                            
                                            st.divider()
                                
                                if text_docs:
                                    with st.expander(f"Text Sources ({len(text_docs)})", expanded=False):
                                        for d in text_docs:
                                            meta = d.metadata
                                            source = meta.get("source", "unknown source")
                                            page = meta.get("page", "?")
                                            
                                            # Show basic source info
                                            st.write(f"**Source:** {source} (page {page})")
                                            
                                            # Show a preview of the text content
                                            content_preview = d.page_content[:200] + "..." if len(d.page_content) > 200 else d.page_content
                                            st.markdown(f"**Content preview:** {content_preview}")
                                            
                                            # Show extraction method if available
                                            if meta.get("extraction_method"):
                                                st.caption(f"Extraction method: {meta['extraction_method']}")
                                            
                                            st.divider()
                                
                                # Save to history
                                st.session_state.history.append((query, answer))
                                
                            except Exception as e:
                                st.error(f"Error processing your question: {str(e)}")
                        else:
                            st.error("No valid image files found. Please re-upload the document.")
                            
                    else:
                        # Text-only answering
                        if text_docs:
                            context_parts = []
                            sources = []
                            
                            for i, doc in enumerate(text_docs):
                                context_parts.append(f"--- Document Section {i+1} ---\n{doc.page_content}")
                                meta = doc.metadata
                                source = meta.get("source", "unknown source")
                                page = meta.get("page", "?")
                                sources.append(f"- {source} (page {page})")
                            
                            combined_context = "\n\n".join(context_parts)
                            
                            prompt = f"""Answer the question based on the following text content from the document(s). 
Analyze ALL the provided text sections to give a thorough and accurate answer.

Context:
{combined_context}

Question: {query}

Please provide a comprehensive answer that synthesizes information from all relevant sections."""
                            
                            selected_model = st.session_state.get('selected_model', DEFAULT_MODEL)
                            answer = call_model_with_retry([prompt])

                            # Display answer
                            st.markdown(f"**Answer:** {answer}")
                            
                            # Show sources in expandable section
                            with st.expander(f"Text Sources ({len(sources)})", expanded=False):
                                for i, (source, doc) in enumerate(zip(sources, text_docs)):
                                    st.write(f"**{source}**")
                                    
                                    # Show a preview of the text content
                                    content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                    st.markdown(f"**Content preview:** {content_preview}")
                                    
                                    # Show extraction method if available
                                    if doc.metadata.get("extraction_method"):
                                        st.caption(f"Extraction method: {doc.metadata['extraction_method']}")
                                    
                                    if i < len(sources) - 1:
                                        st.divider()
                            
                            # Save to history
                            st.session_state.history.append((query, answer))
                        else:
                            st.warning("No relevant documents found for your query. Please try rephrasing your question or upload more documents.")
            else:
                st.error("No vector store available. Please upload documents first.")
        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")
            st.info("Please try rephrasing your question or uploading different documents.")
else:
    st.info("Please upload documents to start asking questions.")

# Conversation History
if st.session_state.get('history'):
    st.markdown("### Conversation History")
    for i, (q, a) in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5 conversations
        with st.expander(f"Q: {q}", expanded=False):
            st.markdown(f"**A:** {a}")
    
    if len(st.session_state.history) > 5:
        st.caption(f"Showing last 5 of {len(st.session_state.history)} conversations")

# How Vision-RAG Works section - moved to bottom
st.markdown("---")
st.markdown("#### How Vision-RAG Works")
with st.expander("Detailed Technical Process", expanded=False):
    st.markdown("""
    ## Document Ingestion Process
    
    **1. File Upload & Detection**
    - Accepts PDF, Word, PowerPoint, and image files (PNG, JPG, JPEG, TIFF)
    - Automatically detects file type and routes to appropriate processor
    
    **2. Text Extraction (Multi-Method Fallback)**
    - **PDFs:** PyMuPDF → pdfminer → UnstructuredPDFLoader (in order of preference)
    - **Word/PowerPoint:** UnstructuredDocumentLoader with format-specific handlers
    - **Images:** Direct vision processing (no text extraction needed)
    
    **3. Image Processing & Vision Analysis**
    - **PDF Images:** Extract embedded images using PyMuPDF with positional data
    - **Standalone Images:** Direct processing of uploaded image files
    - **Context Extraction:** Capture surrounding text near images in documents
    - **Vision Embeddings:** Generate 1024-dimensional vectors using Cohere Embed v4.0
    
    **4. Text Embedding Generation**
    - Convert extracted text to embeddings using Cohere multilingual model
    - Chunk large documents into manageable sections with metadata preservation
    - Generate semantic vectors for similarity search
    
    **5. Vector Store Construction**
    - **Dual Storage:** Separate text and image vector stores for optimal retrieval
    - **Metadata Enrichment:** Store source, page numbers, context, and positioning data
    - **Persistent Storage:** Save processed images to disk for later reference
    
    ---
    
    ## Question Answering & Retrieval Process
    
    **1. Query Processing**
    - Convert user question to embedding vector
    - Generate separate embeddings for text and image search
    
    **2. Similarity Search (Hybrid Approach)**
    - **Text Search:** Cosine similarity against text embeddings
    - **Image Search:** Dot product similarity against vision embeddings
    - **Ranking:** Retrieve top-k results from both stores (default: 6 each)
    
    **3. Context Assembly**
    - **Multi-Modal Combination:** Merge relevant text and image content
    - **Metadata Integration:** Include page numbers, source files, and positioning
    - **Context Enrichment:** Add surrounding text for images when available
    
    **4. AI Model Processing**
    - **Vision-Capable Models:** Process both text and images simultaneously
    - **Text-Only Models:** Process text content with image descriptions
    - **Multi-Provider Support:** OpenAI GPT, Google Gemini, Anthropic Claude
    
    **5. Response Generation**
    - Generate comprehensive answer using retrieved context
    - Cite sources with specific page references
    - Display relevant images and text excerpts
    - Maintain conversation history for follow-up questions
    
    ---
    
    ## Technical Architecture
    
    **Embedding Models:**
    - Text: Cohere multilingual-22-12 (1024 dimensions)
    - Vision: Cohere Embed v4.0 (1024 dimensions)
    
    **Storage Strategy:**
    - Custom vector store with dual text/image indexing
    - Metadata-rich document storage with source tracking
    - Persistent image caching for performance
    
    **Retrieval Strategy:**
    - Hybrid search across text and vision embeddings
    - Contextual ranking with relevance scoring
    - Multi-modal result fusion for comprehensive answers
    """)

# Enhanced styling for clean, streamlined interface
st.markdown("""
<style>
    /* Clean main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Prominent chat input */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 16px;
        font-size: 16px;
        color: #000000 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #45a049;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.2);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #666666;
        font-style: italic;
    }
    
    /* Clean section headers */
    .main h3 {
        color: #2E7D32;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E8F5E8;
    }
    
    /* File uploader styling */
    .stFileUploader {
        margin-bottom: 2rem;
    }
    
    .stFileUploader > div {
        border: 2px dashed #4CAF50;
        border-radius: 8px;
        padding: 1.5rem;
        background-color: transparent !important;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        background-color: rgba(76, 175, 80, 0.05) !important;
    }
    
    /* Handle uploaded file state */
    .stFileUploader section {
        background-color: transparent !important;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stFileUploader section:hover {
        background-color: rgba(76, 175, 80, 0.05) !important;
    }
    
    /* Style uploaded file items */
    .stFileUploader section div {
        background-color: transparent !important;
    }
    
    /* File uploader text styling */
    .stFileUploader label {
        color: #2E7D32 !important;
        font-weight: 600 !important;
    }
    
    /* File uploader drag text */
    .stFileUploader > div > div {
        color: #495057 !important;
        font-weight: 500 !important;
    }
    
    /* Uploaded file names */
    .stFileUploader section div span {
        color: #2E7D32 !important;
        font-weight: 500 !important;
    }
    
    /* File uploader button */
    .stFileUploader section button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
    }
    
    /* File uploader button hover */
    .stFileUploader section button:hover {
        background-color: #45a049 !important;
    }
    
    /* Clean buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    
    /* Clean expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    /* Answer styling */
    .main p strong {
        color: #2E7D32;
        font-size: 1.1rem;
    }
    
    /* Clean success/info messages - Made smaller and more compact */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 6px;
        padding: 0.25rem 0.5rem;
        margin: 0.2rem 0;
        font-size: 0.75rem;
        line-height: 1.2;
    }
    
    /* Make success messages less prominent and smaller */
    .stSuccess {
        background-color: #f8f9fa;
        border: 1px solid #d1e7dd;
        color: #0f5132;
        font-size: 0.75rem;
    }
    
    /* Smaller success icons */
    .stSuccess > div > div > svg {
        width: 12px;
        height: 12px;
    }
    
    /* More compact processing summary */
    .stSuccess small {
        display: block;
        margin: 0.1rem 0;
        color: #495057;
        font-size: 0.7rem;
        line-height: 1.1;
    }
    
    /* Make notification text content smaller */
    .stSuccess > div > div {
        font-size: 0.75rem !important;
        line-height: 1.2 !important;
    }
    
    /* Info messages also smaller */
    .stInfo {
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        line-height: 1.2;
    }
    
    .stInfo > div > div {
        font-size: 0.75rem !important;
        line-height: 1.2 !important;
    }
    
    /* Reduce visual clutter */
    .stSpinner > div {
        border-top-color: #4CAF50;
    }
    
    /* Clean image styling */
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Hide unnecessary elements */
    .css-1rs6os, .css-17eq0hr {
        display: none;
    }
    
    /* Clean dividers */
    hr {
        border: none;
        height: 1px;
        background-color: #e9ecef;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)
