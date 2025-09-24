import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import tempfile
import json
import os
import requests
import re
from datetime import datetime, timedelta
import base64
import io
import pinecone
from pinecone import Pinecone, ServerlessSpec
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fitz  
import docx  
import cv2
from io import BytesIO
import calendar
from collections import defaultdict, Counter
import base64


try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.warning("sentence-transformers not available. Using alternative embedding method.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


st.set_page_config(
    page_title="📄 InvoiceIQ: Intelligent Document Processing Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

image_path = r"C:\Users\SPECTRE\Downloads\assets_task_01k5v8xb75fmcrcahbjpbr9mga_1758630717_img_1.webp"
with open(image_path, "rb") as f:
    data = f.read()
    encoded_image = base64.b64encode(data).decode()


st.markdown(f"""
<style>
/* Set app background */
.stApp {{
    background-image: url("data:image/webp;base64,{encoded_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* Header */
.main-header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}}
.main-header h1 {{
    font-size: 3rem;
    margin-bottom: 0.5rem;
    font-weight: 700;
}}
.main-header p {{
    font-size: 1.2rem;
    opacity: 0.9;
}}

/* Feature cards */
.feature-card {{
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    border: 1px solid #e8ecef;
    text-align: center;
    height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    cursor: pointer;
    margin-bottom: 2rem;
}}
.feature-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
}}
.feature-icon {{
    font-size: 4rem;
    margin-bottom: 1rem;
}}
.feature-title {{
    font-size: 1.5rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 1rem;
}}
.feature-description {{
    color: #7f8c8d;
    font-size: 1rem;
    line-height: 1.6;
    flex-grow: 1;
}}

/* Stats section */
.stats-container {{
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 2rem;
    border-radius: 15px;
    margin: 2rem 0;
    text-align: center;
}}
.stat-item {{
    display: inline-block;
    margin: 0 2rem;
    padding: 1rem;
}}
.stat-number {{
    font-size: 2.5rem;
    font-weight: bold;
    color: #667eea;
    display: block;
}}
.stat-label {{
    color: #6c757d;
    font-size: 1rem;
    margin-top: 0.5rem;
}}

/* Back button */
.back-button {{
    position: fixed;
    top: 20px;
    left: 20px;
    background: #667eea;
    color: white;
    padding: 10px 20px;
    border-radius: 25px;
    border: none;
    cursor: pointer;
    font-weight: 600;
    z-index: 1000;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}}
.back-button:hover {{
    background: #5a67d8;
    transform: translateY(-2px);
}}

/* Section divider */
.section-divider {{
    margin: 3rem 0;
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #667eea, transparent);
}}

/* Analytics header */
.analytics-header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}}

/* Centered text for sections */
.centered-text {{
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)


def show_global_header():
    """Display the global header that appears on all pages"""
    st.markdown("""
    <div class="main-header">
        <h1>📄 InvoiceIQ: Intelligent Document Processing Platform</h1>
        <p>Advanced Multilingual Document Intelligence & Financial Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'invoice_data' not in st.session_state:
    st.session_state.invoice_data = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

@st.cache_resource
def init_pinecone():
    """Initialize Pinecone connection with the new API"""
    try:
        # Get API key from environment variable
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            st.error("PINECONE_API_KEY environment variable not found.")
            return None
        
        # Initialize Pinecone with the new API
        pc = Pinecone(api_key=api_key)
        
        # Create or connect to index
        index_name = "invoice-documents"
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        
        if index_name not in existing_indexes:
            # Create index with appropriate dimension
            dimension = 384 if SENTENCE_TRANSFORMERS_AVAILABLE else 768
            
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            st.sidebar.success(f"Created new Pinecone index: {index_name}")
        
        # Connect to index
        index = pc.Index(index_name)
        return index
        
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

# Initialize embedding model with fallback
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings or use fallback"""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            return None
    else:
        st.warning("Using fallback embedding method. Install sentence-transformers for better results.")
        return None

# Simple fallback embedding function
def simple_embedding(text, dimension=768):
    """Simple fallback embedding function when sentence-transformers is not available"""
    import hashlib
    import numpy as np
    
    # Create a simple deterministic embedding based on text hash
    text_hash = hashlib.md5(text.encode()).hexdigest()
    seed = int(text_hash[:8], 16)  # Use first 8 characters as seed
    
    np.random.seed(seed)
    embedding = np.random.rand(dimension).tolist()
    return embedding

def show_home_page():
    """Display the main home page with three feature sections"""
    show_global_header()
    # Stats section (if there's data)
    if index:
        invoice_data = get_all_invoice_data(index)
        if invoice_data:
            total_docs = len(invoice_data)
            total_suppliers = len(set([inv.get('metadata', {}).get('supplier', '') for inv in invoice_data if inv.get('metadata', {}).get('supplier', '') != 'Unknown']))
            
            st.markdown(f"""
            <div class="stats-container">
                <div class="stat-item">
                    <span class="stat-number">{total_docs}</span>
                    <div class="stat-label">Documents Processed</div>
                </div>
                <div class="stat-item">
                    <span class="stat-number">10</span>
                    <div class="stat-label">Languages Supported</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Three main feature sections
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div>
                <div class="feature-icon">📋</div>
                <div class="feature-title">Stored Documents</div>
                <div class="feature-description">
                    Access and manage all your processed documents. View detailed information, 
                    search through invoices, and organize your document database with ease.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📋 View Stored Documents", key="stored-data-btn", use_container_width=True):
            st.session_state.current_page = 'stored_data'
            st.rerun()

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div>
                <div class="feature-icon">📊</div>
                <div class="feature-title">Financial Dashboard</div>
                <div class="feature-description">
                    Comprehensive analytics and insights from your financial documents. 
                    Track spending, analyze supplier relationships, and visualize financial trends.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Open Dashboard", key="dashboard-btn", use_container_width=True):
            st.session_state.current_page = 'dashboard'
            st.rerun()

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div>
                <div class="feature-icon">🔍</div>
                <div class="feature-title">OCR Processing</div>
                <div class="feature-description">
                    Upload and process new documents with advanced OCR technology. 
                    Extract text from images, PDFs, and Word documents in multiple languages.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Process Documents", key="ocr-btn", use_container_width=True):
            st.session_state.current_page = 'ocr'
            st.rerun()

    # Additional information section
    st.markdown("""
    <style>
        .feature-card {
            background: white;
            color: #333333;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0,0,0,0.10);
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 25px rgba(0,0,0,0.20);
        }
        .feature-icon {
            font-size: 2rem;
        }
        .feature-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-top: 0.5rem;
            color: #111111;
        }
        .feature-description {
            font-size: 0.95rem;
            color: #555555;
            margin-top: 0.5rem;
        }
        .feature-subheader {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            color: #222222;
        }
        .feature-point {
            margin: 0.3rem 0;
            font-size: 0.9rem;
            color: #444444;
        }
    </style>
    """, unsafe_allow_html=True)

    # Centered layout with 4 equal columns
    outer_left, outer_center, outer_right = st.columns([0.2, 4, 0.2])

    with outer_center:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-subheader">🌍 Multilingual Support</div>
                <div class="feature-point">✓ Process documents in 10 languages</div>
                <div class="feature-point">✓ Advanced Hindi text recognition</div>
                <div class="feature-point">✓ Smart language detection</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-subheader">🤖 AI-Powered Analysis</div>
                <div class="feature-point">✓ Intelligent document categorization</div>
                <div class="feature-point">✓ Automatic data extraction</div>
                <div class="feature-point">✓ Financial insights generation</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-subheader">💾 Secure Storage</div>
                <div class="feature-point">✓ Vector database integration</div>
                <div class="feature-point">✓ Search and retrieval system</div>
                <div class="feature-point">✓ Data persistence across sessions</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-subheader">📈 Advanced Analytics</div>
                <div class="feature-point">✓ Real-time dashboard updates</div>
                <div class="feature-point">✓ Supplier relationship mapping</div>
                <div class="feature-point">✓ Financial trend analysis</div>
            </div>
            """, unsafe_allow_html=True)



def show_stored_data_page():
    """Show stored data page with back button"""
    show_global_header()
    st.markdown("""
    <button class="back-button" onclick="document.getElementById('back-home-1').click()">
        ← Back to Home
    </button>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="back-home-1", help="Return to home page"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    show_all_stored_data()

def show_dashboard_page():
    """Show dashboard page with back button"""
    show_global_header()
    st.markdown("""
    <button class="back-button" onclick="document.getElementById('back-home-2').click()">
        ← Back to Home
    </button>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="back-home-2", help="Return to home page"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    create_analytics_dashboard()

def show_ocr_page():
    """Show OCR processing page with back button"""
    show_global_header()
    st.markdown("""
    <button class="back-button" onclick="document.getElementById('back-home-3').click()">
        ← Back to Home
    </button>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="back-home-3", help="Return to home page"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    # Language selection
    st.subheader("🔧 Configuration")
    selected_lang_name = st.selectbox(
        "Document Language",
        options=list(language_options.keys()),
        index=0,
        help="Select the language of the document"
    )
    selected_lang_code = language_options[selected_lang_name]
    
    # File upload section
    st.subheader("📁 Upload Documents")
    st.write("Supported formats: PNG, JPG, JPEG, BMP, PDF, DOCX")
    
    uploaded_files = st.file_uploader(
        "Choose files to process",
        type=['png', 'jpg', 'jpeg', 'bmp', 'pdf', 'docx'],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.subheader("📋 File Processing Queue")
        
        # Display uploaded files
        for i, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name} ({file.type})"):
                st.write(f"**Size:** {file.size:,} bytes")
                st.write(f"**Type:** {file.type}")
                
                if file.type.startswith('image/'):
                    image = Image.open(file)
                    st.image(image, caption=file.name, use_column_width=True)
        
        # Process button
        if st.button("🚀 Process", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for file_idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((file_idx) / len(uploaded_files))
                
                try:
                    if file.type.startswith('image/'):
                        # Process image files
                        image = Image.open(file)
                        
                        if selected_lang_name == "Hindi":
                            texts, confidences, coordinates = llm_ocr_hindi(image)
                        else:
                            ocr = load_ocr(selected_lang_code)
                            img_array = np.array(image.convert('RGB'))
                            result = ocr.ocr(img_array)
                            texts, confidences, coordinates = extract_text_and_coordinates(result)
                        
                        page_info = [1] * len(texts) if texts else []
                    
                    elif file.type == 'application/pdf':
                        # Process PDF files
                        images = extract_images_from_pdf(file)
                        if images:
                            texts, confidences, coordinates, page_info = process_multiple_pages(
                                images, selected_lang_code, selected_lang_name
                            )
                        else:
                            st.error(f"Could not extract images from PDF: {file.name}")
                            continue
                    
                    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                        # Process Word documents
                        text_content = extract_text_from_word(file)
                        if text_content:
                            texts = [text_content]
                            confidences = [1.0]
                            coordinates = [[]]
                            page_info = [1]
                        else:
                            st.error(f"Could not extract text from Word document: {file.name}")
                            continue
                    
                    else:
                        st.error(f"Unsupported file type: {file.type}")
                        continue
                    
                    if texts:
                        # Create results container
                        with st.container():
                            st.success(f"✅ Successfully processed {file.name}")
                            st.write(f"**Extracted {len(texts)} text elements**")
                            
                            # Generate summary
                            formatted_text = format_coordinates_for_llm(texts, coordinates)
                            with st.spinner(f"Generating AI summary for {file.name}..."):
                                summary = summarize_with_deepseek(formatted_text, coordinates)
                            
                            if summary:
                                st.subheader(f"🤖 AI Analysis - {file.name}")
                                st.write(summary)
                                
                                # Extract metadata and store
                                metadata = extract_metadata_from_summary(summary)
                                metadata["filename"] = file.name
                                metadata["file_type"] = file.type
                                metadata["pages"] = max(page_info) if page_info else 1
                                metadata["text_elements"] = len(texts)
                                
                                # Store in Pinecone
                                if index:
                                    store_success = store_in_pinecone(
                                        formatted_text, 
                                        summary, 
                                        metadata, 
                                        index, 
                                        embedding_model
                                    )
                                    if store_success:
                                        st.session_state.processed_files.append(file.name)
                                
                                # Download options
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label=f"📝 Download Summary",
                                        data=summary,
                                        file_name=f"{file.name}_summary.txt",
                                        mime="text/plain"
                                    )
                                
                                with col2:
                                    json_output = {
                                        "filename": file.name,
                                        "metadata": metadata,
                                        "extracted_text": texts,
                                        "confidences": confidences,
                                        "summary": summary
                                    }
                                    st.download_button(
                                        label=f"💾 Download JSON",
                                        data=json.dumps(json_output, indent=2, ensure_ascii=False),
                                        file_name=f"{file.name}_data.json",
                                        mime="application/json"
                                    )
                            
                            st.divider()
                    
                    else:
                        st.warning(f"⚠️ No text detected in {file.name}")
                
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                
                # Update progress
                progress_bar.progress((file_idx + 1) / len(uploaded_files))
            
            status_text.text("✅ All files processed!")

# Language selection
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Chinese": "ch",
    "French": "fr",
    "German": "german",
    "Korean": "korean",
    "Japanese": "japan",
    "Spanish": "es",
    "Portuguese": "pt",
    "Russian": "ru"
}

# Load OCR model
@st.cache_resource
def load_ocr(lang_code):
    return PaddleOCR(lang=lang_code, use_angle_cls=True)

# PDF processing functions
def extract_images_from_pdf(pdf_file):
    """Extract images from each page of PDF"""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # Convert page to image
        mat = fitz.Matrix(2, 2)  # Increase resolution
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    
    pdf_document.close()
    return images

def extract_text_from_word(docx_file):
    """Extract text from Word document"""
    doc = docx.Document(docx_file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def process_multiple_pages(images, selected_lang_code, selected_lang_name):
    """Process multiple pages and return combined results"""
    all_texts = []
    all_confidences = []
    all_coordinates = []
    page_info = []
    
    for page_num, image in enumerate(images):
        st.write(f"Processing page {page_num + 1}...")
        
        if selected_lang_name == "Hindi":
            texts, confidences, coordinates = llm_ocr_hindi(image)
        else:
            ocr = load_ocr(selected_lang_code)
            img_array = np.array(image.convert('RGB'))
            result = ocr.ocr(img_array)
            texts, confidences, coordinates = extract_text_and_coordinates(result)
        
        if texts:
            # Add page information to each text element
            page_texts = [f"[Page {page_num + 1}] {text}" for text in texts]
            all_texts.extend(page_texts)
            all_confidences.extend(confidences)
            all_coordinates.extend(coordinates)
            page_info.extend([page_num + 1] * len(texts))
    
    return all_texts, all_confidences, all_coordinates, page_info

# Original OCR and processing functions
def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def llm_ocr_hindi(image):
    """Use LLM for OCR when Hindi is detected"""
    try:
        # Check if API key is available
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            st.error("OPENROUTER_API_KEY environment variable not found. Please set it before running the app.")
            return None, None, None
        
        # Convert image to base64
        base64_image = image_to_base64(image)
        
        # Enhanced prompt for Hindi OCR
        prompt = """
        Please perform OCR on this image and extract all visible text. The image contains Hindi text.
        
        INSTRUCTIONS:
        1. Extract ALL visible text from the image, including Hindi/Devanagari script
        2. Maintain the original text in its native script (Hindi/Devanagari)
        3. For each text element, provide:
           - The extracted text (in original Hindi script)
           - Approximate position information (describe location like "top-left", "center", "bottom-right", etc.)
           - Confidence level (estimate from 0.0 to 1.0)
        
        4. Format your response as a JSON structure:
        {
            "texts": ["text1", "text2", ...],
            "positions": ["top-left", "center", ...],
            "confidences": [0.95, 0.87, ...]
        }
        
        5. If you see any English text mixed with Hindi, include that as well
        6. Preserve all numbers, symbols, and punctuation marks exactly as they appear
        7. If text is unclear or partially visible, include it with a lower confidence score
        
        Please analyze the image and provide the OCR results in the specified JSON format.
        """
        
        # OpenRouter API request with vision capability
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "openai/gpt-4o",  # Using GPT-4O for vision capabilities
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60  # 60 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_result = json.loads(json_str)
                    
                    texts = parsed_result.get('texts', [])
                    positions = parsed_result.get('positions', [])
                    confidences = parsed_result.get('confidences', [])
                    
                    # Create dummy coordinates based on position descriptions
                    coordinates = []
                    for i, pos in enumerate(positions):
                        # Create approximate coordinates based on position description
                        if 'top' in pos.lower() and 'left' in pos.lower():
                            coord = [[100, 100], [300, 100], [300, 150], [100, 150]]
                        elif 'top' in pos.lower() and 'right' in pos.lower():
                            coord = [[400, 100], [600, 100], [600, 150], [400, 150]]
                        elif 'bottom' in pos.lower() and 'left' in pos.lower():
                            coord = [[100, 400], [300, 400], [300, 450], [100, 450]]
                        elif 'bottom' in pos.lower() and 'right' in pos.lower():
                            coord = [[400, 400], [600, 400], [600, 450], [400, 450]]
                        elif 'center' in pos.lower():
                            coord = [[250, 250], [450, 250], [450, 300], [250, 300]]
                        elif 'top' in pos.lower():
                            coord = [[200, 100], [400, 100], [400, 150], [200, 150]]
                        elif 'bottom' in pos.lower():
                            coord = [[200, 400], [400, 400], [400, 450], [200, 450]]
                        elif 'left' in pos.lower():
                            coord = [[100, 200], [300, 200], [300, 250], [100, 250]]
                        elif 'right' in pos.lower():
                            coord = [[400, 200], [600, 200], [600, 250], [400, 250]]
                        else:
                            # Default center position with slight offset for each item
                            y_offset = i * 50 + 200
                            coord = [[200, y_offset], [400, y_offset], [400, y_offset + 40], [200, y_offset + 40]]
                        
                        coordinates.append(coord)
                    
                    # Ensure all lists are the same length
                    min_len = min(len(texts), len(confidences))
                    texts = texts[:min_len]
                    confidences = confidences[:min_len]
                    coordinates = coordinates[:min_len] if coordinates else [[]] * min_len
                    
                    return texts, confidences, coordinates
                else:
                    # If no JSON found, try to parse the raw text
                    st.warning("Could not parse JSON from LLM response. Using raw text.")
                    lines = content.strip().split('\n')
                    texts = [line.strip() for line in lines if line.strip()]
                    confidences = [0.8] * len(texts)  # Default confidence
                    coordinates = [[]] * len(texts)  # Empty coordinates
                    return texts, confidences, coordinates
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, use the raw response
                st.warning("Could not parse structured response from LLM. Using raw text.")
                lines = content.strip().split('\n')
                texts = [line.strip() for line in lines if line.strip()]
                confidences = [0.8] * len(texts)  # Default confidence
                coordinates = [[]] * len(texts)  # Empty coordinates
                return texts, confidences, coordinates
        else:
            st.error(f"OpenRouter API error: {response.status_code} - {response.text}")
            return None, None, None
        
    except Exception as e:
        st.error(f"Error in LLM OCR: {str(e)}")
        return None, None, None

def extract_text_and_coordinates(result):
    """Extract text and coordinates from PaddleOCR result with robust handling"""
    texts = []
    confidences = []
    coordinates = []
    
    if not result:
        return texts, confidences, coordinates
    
    try:
        # Handle different possible result structures
        for page in result:
            if page is None:
                continue
                
            # Check if this is a new format with rec_texts attribute
            if hasattr(page, 'rec_texts'):
                # New format: OCRResult object
                for i, text in enumerate(page.rec_texts):
                    if text and hasattr(page, 'rec_scores') and i < len(page.rec_scores):
                        texts.append(text)
                        confidences.append(page.rec_scores[i])
                        if hasattr(page, 'rec_boxes') and i < len(page.rec_boxes):
                            # Convert to list if it's a numpy array
                            coords = page.rec_boxes[i]
                            if hasattr(coords, 'tolist'):
                                coordinates.append(coords.tolist())
                            else:
                                coordinates.append(coords)
            
            # Check if this is a dictionary format
            elif isinstance(page, dict) and 'rec_texts' in page:
                # Dictionary format
                for i, text in enumerate(page['rec_texts']):
                    if text and 'rec_scores' in page and i < len(page['rec_scores']):
                        texts.append(text)
                        confidences.append(page['rec_scores'][i])
                        if 'rec_boxes' in page and i < len(page['rec_boxes']):
                            coords = page['rec_boxes'][i]
                            if hasattr(coords, 'tolist'):
                                coordinates.append(coords.tolist())
                            else:
                                coordinates.append(coords)
            
            # Traditional list format
            elif isinstance(page, list):
                for line in page:
                    if line is None:
                        continue
                    
                    # Handle different line formats
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        # Format: [coordinates, (text, confidence)]
                        if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                            text = str(line[1][0])
                            confidence = float(line[1][1])
                            coords = line[0]
                            texts.append(text)
                            confidences.append(confidence)
                            # Convert to list if it's a numpy array
                            if hasattr(coords, 'tolist'):
                                coordinates.append(coords.tolist())
                            else:
                                coordinates.append(coords)
                        
                        # Format: [coordinates, text, confidence]
                        elif len(line) >= 3:
                            text = str(line[1])
                            confidence = float(line[2])
                            coords = line[0]
                            texts.append(text)
                            confidences.append(confidence)
                            # Convert to list if it's a numpy array
                            if hasattr(coords, 'tolist'):
                                coordinates.append(coords.tolist())
                            else:
                                coordinates.append(coords)
    
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        # Fallback: try to extract any text we can find
        try:
            # Convert the entire result to string and look for text patterns
            result_str = str(result)
            # Simple pattern to find text in quotes (common in OCR results)
            text_matches = re.findall(r"'(.*?)'", result_str)
            texts.extend(text_matches)
            confidences.extend([0.5] * len(text_matches))  # Default confidence
            coordinates.extend([[]] * len(text_matches))  # Empty coordinates
        except:
            pass
    
    return texts, confidences, coordinates

def format_coordinates_for_llm(texts, coordinates):
    """Format text with coordinates for LLM analysis"""
    if not coordinates or len(texts) != len(coordinates):
        return "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    
    formatted_output = []
    for i, (text, coord) in enumerate(zip(texts, coordinates)):
        # Handle different coordinate formats
        if coord and isinstance(coord, (list, tuple)) and len(coord) > 0:
            # Check if we have a list of points
            if isinstance(coord[0], (list, tuple)) and len(coord[0]) >= 2:
                # Calculate center point of the bounding box
                try:
                    x_coords = [point[0] for point in coord]
                    y_coords = [point[1] for point in coord]
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    formatted_output.append(f"{i+1}. [{center_x:.1f},{center_y:.1f}] {text}")
                except (TypeError, IndexError):
                    formatted_output.append(f"{i+1}. {text}")
            else:
                formatted_output.append(f"{i+1}. {text}")
        else:
            formatted_output.append(f"{i+1}. {text}")
    
    return "\n".join(formatted_output)

def summarize_with_deepseek(text_content, coordinate_data):
    """Summarize extracted text using DeepSeek v3 via OpenRouter API with coordinate information"""
    try:
        # Check if API key is available
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            st.error("OPENROUTER_API_KEY environment variable not found. Please set it before running the app.")
            return None
        
        # Truncate text to fit within token limits (approx 6000 characters for safety)
        max_chars = 6000
        if len(text_content) > max_chars:
            truncated_text = text_content[:max_chars] + "... [text truncated due to length]"
            st.warning(f"Text truncated from {len(text_content)} to {max_chars} characters for summarization")
        else:
            truncated_text = text_content
        
        # Enhanced prompt with coordinate information
        prompt = f"""
        Analyze this business document text with coordinate information and provide a complete English translation while extracting all information.

        CRITICAL RULES:
        1. TRANSLATE ALL TEXT to English - every word, label, description, and note
        2. PRESERVE numbers, codes, currencies, and values exactly as they appear
        3. USE COORDINATE INFORMATION [x,y] to understand document layout and relationships
        4. TEXT WITH SIMILAR Y-COORDINATES is likely on the same line
        5. TEXT WITH SIMILAR X-COORDINATES is likely in the same column
        6. EXTRACT only visible information - no assumptions or guessing
        7. TRANSLATE COMPLETELY - ensure no text remains in the original language
        8. ***CRITICAL ENTITY SEPARATION:***
            - When text appears on SEPARATE LINES (different Y-coordinates), treat as SEPARATE ENTITIES
            - When clear visual SPACING exists between text blocks, treat as SEPARATE ENTITIES
            - DO NOT COMBINE entities that are visually separated
        9. ***CRITICAL ROLE ASSIGNMENT:***
            - The entity with CONTACT INFORMATION (phone, email, website) is the SELLER
            - The entity with only an ADDRESS is typically the BUYER
            - The FIRST entity appearing in the document (lowest Y-coordinate) is typically the SELLER
            - The entity that appears AFTER the seller is typically the BUYER
        10. Look for explicit labels like "Seller:", "Buyer:", "From:", "To:" to identify relationships
        11. If no labels are present, use document positioning (seller typically appears first/above buyer)
        12. ***CRITICAL FOR DATES: Convert ALL dates to DD/MM/YYYY format. This is mandatory.***
        13. If the original text uses a different format (e.g., YYYY/MM/DD, MM-DD-YYYY), you MUST convert it to DD/MM/YYYY.
        14. ***SMART TRANSLATION APPROACH:***
            - First identify the language of the document
            - Recognize common abbreviation patterns in that language
            - Single characters with symbols (like H°, N°) usually mean "Number"
            - Translate abbreviations based on their function, not just literal meaning
            - Convert non-English numbers to their English equivalents
            - Preserve the original structure while making it understandable in English
        15. ***CONTEXTUAL UNDERSTANDING:***
            - If you see a pattern like "[Word] H°: [value]" → It usually means "[Word] Number: [value]"
            - If you see a pattern like "[Word]: [value]" → Translate both the label and the value
            - Use coordinate positioning to understand relationships between labels and values
        16. ***DATE CALCULATION RULES:***
            - If payment terms specify a period after delivery (e.g., "within X days after delivery")
            - Calculate the due date by adding X days to the delivery date
            - Use the format: Delivery Date + X days = Due Date (DD/MM/YYYY)
            - Show your calculation in the notes section
            - If no delivery date is provided, state "Due date cannot be calculated without delivery date"
        17. ***EXPLICIT CALCULATION REQUIREMENT:***
            - You MUST calculate due dates when payment terms provide enough information
            - You MUST show your calculation logic
            - Never leave due date as "N/A" when it can be calculated from available information

        TEXT WITH COORDINATES (format: [x,y] text):
        {truncated_text}

        COMPLETE ENGLISH TRANSLATION AND INFORMATION EXTRACTION:

        DOCUMENT TYPE: 

        SELLER INFORMATION (the entity providing goods/services):
        - Name: 
        - Address: 
        - Contact: 
        - Tax ID: 

        BUYER INFORMATION (the entity purchasing goods/services):
        - Name: 
        - Address: 

        DOCUMENT DETAILS:
        - Document Number: 
        - Order Number:
        - Date: 
        - Delivery Date: 
        - Due Date: 

        PRODUCTS/SERVICES:
        - Item:
        - Description: 
        - Quantity: 
        - Unit Price: 
        - Total Price: 

        FINANCIAL INFORMATION:
        - Subtotal: 
        - Tax: 
        - Grand Total: 
        - Amount Paid: 
        - Balance Due: 

        PAYMENT TERMS: 

        NOTES: 
        """
        
        # OpenRouter API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek/deepseek-chat-v3",  # OpenRouter model name
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2500
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60  # 60 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result['choices'][0]['message']['content']
            return summary
        else:
            st.error(f"OpenRouter API error: {response.status_code} - {response.text}")
            return None
        
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return None

def extract_metadata_from_summary(summary):
    """Extract structured metadata from the LLM summary with improved parsing"""
    metadata = {
        "supplier": "Unknown",
        "buyer": "Unknown",
        "date": "Unknown",
        "total": "Unknown",
        "document_type": "Unknown",
        "document_number": "Unknown",
        "tax_amount": "Unknown",
        "subtotal": "Unknown",
        "currency": "Unknown",
        "summary": summary[:500] + "..." if len(summary) > 500 else summary,
        "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        # Extract supplier - multiple patterns with improved parsing
        supplier_patterns = [
            r"SELLER INFORMATION[^:]*:.*?Name:\s*(.*?)(?:\n|Address:|Contact:|$)",
            r"SELLER[^:]*:.*?Name:\s*(.*?)(?:\n|$)",
            r"Seller:\s*(.*?)(?:\n|$)",
            r"Verkäufer:\s*(.*?)(?:\n|$)",
            r"発注者:\s*(.*?)(?:\n|$)",
            r"销售方:\s*(.*?)(?:\n|$)",
            r"SELLER[^:]*:\s*(.*?)(?:\n|Address:|$)",
            r"From:\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in supplier_patterns:
            supplier_match = re.search(pattern, summary, re.IGNORECASE | re.DOTALL)
            if supplier_match and supplier_match.group(1).strip() not in ["", "Not explicitly provided", "Not provided", "N/A"]:
                supplier_name = supplier_match.group(1).strip()
                # Clean up the extracted name
                supplier_name = re.sub(r'^\W+|\W+$', '', supplier_name)  # Remove leading/trailing non-word chars
                if supplier_name and len(supplier_name) > 2:  # Ensure it's a valid name
                    metadata["supplier"] = supplier_name
                    break
        
        # Extract buyer - multiple patterns with improved parsing
        buyer_patterns = [
            r"BUYER INFORMATION[^:]*:.*?Name:\s*(.*?)(?:\n|Address:|$)",
            r"BUYER[^:]*:.*?Name:\s*(.*?)(?:\n|$)",
            r"Buyer:\s*(.*?)(?:\n|$)",
            r"Käufer:\s*(.*?)(?:\n|$)",
            r"納入先:\s*(.*?)(?:\n|$)",
            r"购买方:\s*(.*?)(?:\n|$)",
            r"BUYER[^:]*:\s*(.*?)(?:\n|Address:|$)",
            r"To:\s*(.*?)(?:\n|$)",
            r"Bill To:\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in buyer_patterns:
            buyer_match = re.search(pattern, summary, re.IGNORECASE | re.DOTALL)
            if buyer_match and buyer_match.group(1).strip() not in ["", "Not explicitly provided", "Not provided", "N/A"]:
                buyer_name = buyer_match.group(1).strip()
                buyer_name = re.sub(r'^\W+|\W+$', '', buyer_name)
                if buyer_name and len(buyer_name) > 2:
                    metadata["buyer"] = buyer_name
                    break
        
        # Extract date with multiple patterns
        date_patterns = [
            r"Date:\s*(.*?)(?:\n|$)",
            r"Rechnungsdatum:\s*(.*?)(?:\n|$)",
            r"発注日:\s*(.*?)(?:\n|$)",
            r"开票日期:\s*(.*?)(?:\n|$)",
            r"Invoice Date:\s*(.*?)(?:\n|$)",
            r"Date de facturation:\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, summary, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1).strip()
                if date_str not in ["", "Not provided", "N/A", "Unknown"]:
                    # Clean date string
                    date_str = re.sub(r'[^\d/\-\.]', '', date_str)  # Remove non-date characters
                    if date_str and len(date_str) > 4:  # Ensure it's a valid date
                        metadata["date"] = date_str
                        break
        
        # Extract total with improved currency detection and parsing
        total_patterns = [
            r"Grand Total:\s*([^\n]+)(?:\n|$)",
            r"Total:\s*([^\n]+)(?:\n|$)",
            r"合計金額:\s*([^\n]+)(?:\n|$)",
            r"Gesamtbetrag:\s*([^\n]+)(?:\n|$)",
            r"Summe:\s*([^\n]+)(?:\n|$)"
        ]
        
        for pattern in total_patterns:
            total_match = re.search(pattern, summary, re.IGNORECASE)
            if total_match:
                total_text = total_match.group(1).strip()
                if total_text not in ["", "Not provided", "N/A", "Unknown"]:
                    # Clean and extract numeric value
                    clean_total = re.sub(r'[^\d\.,]', '', total_text)
                    if clean_total:
                        # Handle different decimal formats
                        clean_total = clean_total.replace(',', '.')
                        if clean_total.count('.') > 1:
                            # Handle thousand separators (e.g., 1,000.00 -> 1000.00)
                            clean_total = clean_total.replace('.', '', clean_total.count('.') - 1)
                        
                        try:
                            numeric_value = float(clean_total)
                            metadata["total"] = f"{numeric_value:,.2f}"
                        except ValueError:
                            metadata["total"] = total_text
                    
                    # Detect currency
                    currency_symbols = {
                        '€': 'EUR', '¥': 'JPY', '$': 'USD', '£': 'GBP', 
                        '₹': 'INR', '元': 'CNY', '￥': 'JPY', 'RMB': 'CNY'
                    }
                    
                    for symbol, currency_code in currency_symbols.items():
                        if symbol in total_text:
                            metadata["currency"] = currency_code
                            break
                    
                    if metadata["currency"] == "Unknown":
                        # Check for currency text
                        if 'euro' in total_text.lower() or 'eur' in total_text.lower():
                            metadata["currency"] = 'EUR'
                        elif 'dollar' in total_text.lower() or 'usd' in total_text.lower():
                            metadata["currency"] = 'USD'
                        elif 'yen' in total_text.lower() or 'jpy' in total_text.lower():
                            metadata["currency"] = 'JPY'
                        elif 'pound' in total_text.lower() or 'gbp' in total_text.lower():
                            metadata["currency"] = 'GBP'
                        elif 'yuan' in total_text.lower() or 'cny' in total_text.lower():
                            metadata["currency"] = 'CNY'
                        elif 'rupee' in total_text.lower() or 'inr' in total_text.lower():
                            metadata["currency"] = 'INR'
        
        # Extract document type
        doc_type_patterns = [
            r"DOCUMENT TYPE:\s*(.*?)(?:\n|$)",
            r"Type de document:\s*(.*?)(?:\n|$)",
            r"ドキュメントタイプ:\s*(.*?)(?:\n|$)",
            r"文档类型:\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in doc_type_patterns:
            doc_type_match = re.search(pattern, summary, re.IGNORECASE)
            if doc_type_match:
                doc_type = doc_type_match.group(1).strip()
                if doc_type not in ["", "Not provided", "N/A"]:
                    metadata["document_type"] = doc_type
                    break
        
        # Extract document number with multiple patterns
        doc_num_patterns = [
            r"Document Number:\s*(.*?)(?:\n|$)",
            r"Rechnungs-Nr\.:\s*(.*?)(?:\n|$)",
            r"发票号码:\s*(.*?)(?:\n|$)",
            r"請求書番号:\s*(.*?)(?:\n|$)",
            r"Invoice #:\s*(.*?)(?:\n|$)",
            r"No\.:\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in doc_num_patterns:
            doc_num_match = re.search(pattern, summary, re.IGNORECASE)
            if doc_num_match:
                doc_num = doc_num_match.group(1).strip()
                if doc_num not in ["", "Not provided", "N/A", "Unknown"]:
                    metadata["document_number"] = doc_num
                    break
        
        # Extract tax amount
        tax_patterns = [
            r"Tax[^:]*:\s*([^\n]+)(?:\n|$)",
            r"Tax Amount:\s*([^\n]+)(?:\n|$)",
            r"税额:\s*([^\n]+)(?:\n|$)",
            r"Steuer:\s*([^\n]+)(?:\n|$)",
            r"消費税:\s*([^\n]+)(?:\n|$)"
        ]
        
        for pattern in tax_patterns:
            tax_match = re.search(pattern, summary, re.IGNORECASE)
            if tax_match:
                tax_text = tax_match.group(1).strip()
                if tax_text not in ["", "Not provided", "N/A", "Unknown"]:
                    # Clean and extract numeric value
                    clean_tax = re.sub(r'[^\d\.,]', '', tax_text)
                    if clean_tax:
                        clean_tax = clean_tax.replace(',', '.')
                        if clean_tax.count('.') > 1:
                            clean_tax = clean_tax.replace('.', '', clean_tax.count('.') - 1)
                        try:
                            numeric_tax = float(clean_tax)
                            metadata["tax_amount"] = f"{numeric_tax:,.2f}"
                        except ValueError:
                            metadata["tax_amount"] = tax_text
                    break
        
        # Extract subtotal
        subtotal_patterns = [
            r"Subtotal:\s*([^\n]+)(?:\n|$)",
            r"Zwischensumme:\s*([^\n]+)(?:\n|$)",
            r"小計:\s*([^\n]+)(?:\n|$)",
            r"金額:\s*([^\n]+)(?:\n|$)"
        ]
        
        for pattern in subtotal_patterns:
            subtotal_match = re.search(pattern, summary, re.IGNORECASE)
            if subtotal_match:
                subtotal_text = subtotal_match.group(1).strip()
                if subtotal_text not in ["", "Not provided", "N/A", "Unknown"]:
                    clean_subtotal = re.sub(r'[^\d\.,]', '', subtotal_text)
                    if clean_subtotal:
                        clean_subtotal = clean_subtotal.replace(',', '.')
                        if clean_subtotal.count('.') > 1:
                            clean_subtotal = clean_subtotal.replace('.', '', clean_subtotal.count('.') - 1)
                        try:
                            numeric_subtotal = float(clean_subtotal)
                            metadata["subtotal"] = f"{numeric_subtotal:,.2f}"
                        except ValueError:
                            metadata["subtotal"] = subtotal_text
                    break
                    
    except Exception as e:
        st.warning(f"Could not extract all metadata from summary: {e}")
        # Fallback: try to find basic information using simpler patterns
        try:
            # Look for any company names in the summary
            company_pattern = r'[A-Z][a-zA-Z\s&\.\,]+(?:GmbH|Co\.|Ltd|Inc|Corp|LLC|Pvt|AG)'
            companies = re.findall(company_pattern, summary)
            if companies and len(companies) >= 2:
                metadata["supplier"] = companies[0]
                metadata["buyer"] = companies[1]
            elif companies and len(companies) == 1:
                metadata["supplier"] = companies[0]
            
            # Look for dates in various formats
            date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'
            dates = re.findall(date_pattern, summary)
            if dates:
                metadata["date"] = dates[0]
                
            # Look for currency amounts
            amount_pattern = r'[€¥$£₹]\s*[\d,]+\.?\d*'
            amounts = re.findall(amount_pattern, summary)
            if amounts:
                metadata["total"] = amounts[-1]  # Use the last amount (usually grand total)
                
        except:
            pass  # If fallback also fails, keep the default values
    
    return metadata


def store_in_pinecone(text_content, summary, metadata, index, embedding_model):
    """Store the invoice data in Pinecone vector database"""
    try:
        # Generate embedding for the text content
        if embedding_model:
            # Use sentence-transformers if available
            embedding = embedding_model.encode(text_content).tolist()
        else:
            # Use fallback embedding method
            embedding = simple_embedding(text_content)
        
        # Generate a unique ID
        doc_id = str(uuid.uuid4())
        
        # Prepare metadata for Pinecone
        pinecone_metadata = {
            "text": text_content[:1000] + "..." if len(text_content) > 1000 else text_content,
            "full_summary": summary,
            **metadata
        }
        
        # Upsert to Pinecone
        index.upsert([
            (doc_id, embedding, pinecone_metadata)
        ])
        
        # Store in session state as well
        st.session_state.invoice_data.append({
            "id": doc_id,
            "metadata": metadata,
            "summary": summary,
            "text": text_content
        })
        
        st.success(f"✅ Invoice stored in Pinecone with ID: {doc_id}")
        return True
        
    except Exception as e:
        st.error(f"Error storing in Pinecone: {str(e)}")
        return False

def get_all_invoice_data(index):
    """Retrieve all invoice data from Pinecone"""
    try:
        # Query Pinecone for all documents
        results = index.query(
            vector=[0.1] * (384 if SENTENCE_TRANSFORMERS_AVAILABLE else 768),
            top_k=1000,
            include_metadata=True,
            include_values=False
        )
        
        invoice_data = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            invoice_data.append({
                "id": match.get('id', ''),
                "metadata": metadata,
                "summary": metadata.get('full_summary', ''),
                "text": metadata.get('text', '')
            })
        
        return invoice_data
        
    except Exception as e:
        st.error(f"Error retrieving invoice data from Pinecone: {str(e)}")
        return []

def create_analytics_dashboard():
    """Create comprehensive analytics dashboard using Pinecone data"""
    if not index:
        st.error("Database not available.")
        return
    
    # Get data from Pinecone
    invoice_data = get_all_invoice_data(index)
    
    if not invoice_data:
        st.warning("No data available for analysis.")
        return
    
    # Prepare clean data
    financial_data = []
    supplier_data = []
    date_data = []
    
    for invoice in invoice_data:
        metadata = invoice.get('metadata', {})
        
        # Extract numeric values without currency symbols
        def extract_numeric_value(value_str):
            if not value_str or value_str == 'Unknown':
                return None
            try:
                # Remove all non-digit characters except decimal point and minus
                clean_value = re.sub(r'[^\d.-]', '', str(value_str))
                if clean_value and clean_value != '.':
                    return float(clean_value)
            except:
                pass
            return None
        
        total_value = extract_numeric_value(metadata.get('total'))
        tax_value = extract_numeric_value(metadata.get('tax_amount'))
        subtotal_value = extract_numeric_value(metadata.get('subtotal'))
        
        if total_value is not None and total_value > 0:
            financial_data.append({
                'supplier': metadata.get('supplier', 'Unknown'),
                'buyer': metadata.get('buyer', 'Unknown'),
                'total': total_value,
                'tax': tax_value if tax_value is not None else 0,
                'subtotal': subtotal_value if subtotal_value is not None else total_value,
                'date': metadata.get('date', 'Unknown'),
                'document_type': metadata.get('document_type', 'Unknown'),
                'document_number': metadata.get('document_number', 'Unknown'),
                'currency': metadata.get('currency', 'Unknown'),
                'id': invoice.get('id', '')
            })
        
        # Collect supplier data
        supplier_data.append({
            'supplier': metadata.get('supplier', 'Unknown'),
            'buyer': metadata.get('buyer', 'Unknown')
        })
        
        # Collect date data
        date_str = metadata.get('date', '')
        if date_str != 'Unknown' and date_str:
            date_data.append({
                'date': date_str,
                'month': date_str.split('/')[1] if '/' in date_str else 'Unknown',
                'year': date_str.split('/')[2] if '/' in date_str and len(date_str.split('/')) > 2 else 'Unknown'
            })
    
    if not financial_data:
        st.info("No valid financial data found.")
        return
    
    df = pd.DataFrame(financial_data)
    supplier_df = pd.DataFrame(supplier_data)
    date_df = pd.DataFrame(date_data)
    
    # Clean UI with minimal styling
    st.markdown("""
    <style>
    .analytics-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
   # Metrics
    total_invoices = len(df)
    handwritten_invoices = 1  # Hardcoded value
    human_intervention = 0     # Replace with your logic if needed

    # Layout with 3 cards side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div style="
                background-color: #4CAF50;
                color: white;
                padding: 30px;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 40px; font-weight: bold;">{total_invoices}</div>
                <div style="font-size: 18px;">Total Invoices</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="
                background-color: #2196F3;
                color: white;
                padding: 30px;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 40px; font-weight: bold;">{handwritten_invoices}</div>
                <div style="font-size: 18px;">Handwritten</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="
                background-color: #FF5722;
                color: white;
                padding: 30px;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 40px; font-weight: bold;">{human_intervention}</div>
                <div style="font-size: 18px;">Human Intervention</div>
            </div>
        """, unsafe_allow_html=True)



    

    
    # Main Charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    

    # Document type analysis
    st.markdown("---")
    st.subheader("📄 Document Types")

    if 'document_type' in df.columns:
        doc_type_counts = df['document_type'].value_counts()
        fig_doc = px.bar(
            x=doc_type_counts.index,
            y=doc_type_counts.values,
            title="Document Types Distribution",
            labels={'x': 'Document Type', 'y': 'Count'}
        )
        st.plotly_chart(fig_doc, use_container_width=True)
    else:
        st.warning("Document type column not found in data.")

    
    # Recent invoices table
    st.markdown("---")
    st.subheader("📋 Recent Invoices")
    
    # Display recent invoices with currency info
    recent_data = df.nlargest(10, 'total')[['supplier', 'total', 'currency', 'date', 'document_number']]
    recent_data['amount_with_currency'] = recent_data.apply(
        lambda x: f"{x['total']:,.0f} {x['currency']}" if x['currency'] != 'Unknown' else f"{x['total']:,.0f}", 
        axis=1
    )
    recent_data = recent_data[['supplier', 'amount_with_currency', 'date', 'document_number']]
    recent_data.columns = ['Supplier', 'Amount', 'Date', 'Document #']
    
    st.dataframe(
        recent_data,
        use_container_width=True,
        height=400
    )
    
    # Detailed statistics
    st.markdown("---")
    st.subheader("📈 Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Financial Summary**")
        st.write(f"- Total amount: {df['total'].sum():,.0f}")
        st.write(f"- Average invoice: {df['total'].mean():,.0f}")
        st.write(f"- Highest invoice: {df['total'].max():,.0f}")
        st.write(f"- Lowest invoice: {df['total'].min():,.0f}")
    
    with col2:
        st.write("**Business Insights**")
        st.write(f"- Unique suppliers: {supplier_df['supplier'].nunique()}")
        st.write(f"- Unique buyers: {supplier_df['buyer'].nunique()}")
        st.write(f"- Document types: {df['document_type'].nunique()}")
        st.write(f"- Currencies used: {df['currency'].nunique()}")
        
    # Supplier-Buyer network
    st.markdown("---")
    st.subheader("🤝 Supplier-Buyer Relationships")
    
    relationship_counts = supplier_df.groupby(['supplier', 'buyer']).size().reset_index(name='count')
    if not relationship_counts.empty:
        st.write(f"**Business Network:** {len(relationship_counts)} unique relationships")
        st.dataframe(
            relationship_counts.sort_values('count', ascending=False).head(10),
            use_container_width=True
        )

def show_all_stored_data():
    """Show all stored data from Pinecone with full details"""
    if not index:
        st.error("Database not available.")
        return
    
    st.header("📋 All Stored Invoices")
    
    # Get all data from Pinecone
    invoice_data = get_all_invoice_data(index)
    
    if not invoice_data:
        st.info("No invoices stored yet.")
        return
    
    # Display each invoice in an expandable card
    for i, invoice in enumerate(invoice_data):
        metadata = invoice.get('metadata', {})
        
        # Create a better title with actual data
        title_parts = []
        if metadata.get('document_number', 'Unknown') != 'Unknown':
            title_parts.append(f"Doc#: {metadata['document_number']}")
        if metadata.get('supplier', 'Unknown') != 'Unknown':
            title_parts.append(f"Supplier: {metadata['supplier']}")
        if metadata.get('buyer', 'Unknown') != 'Unknown':
            title_parts.append(f"Buyer: {metadata['buyer']}")
        
        title = f"📄 Invoice {i+1}: {' | '.join(title_parts)}" if title_parts else f"📄 Invoice {i+1}"
        
        with st.expander(title, expanded=False):
            st.markdown("---")
            st.write("**Full Summary:**")
            st.write(invoice.get('summary', 'No summary available'))         
            st.write("**ID:**", invoice.get('id', 'Unknown'))


# Initialize Pinecone and embedding model
index = init_pinecone()
embedding_model = load_embedding_model()

# Main app routing
if st.session_state.current_page == 'home':
    show_home_page()
elif st.session_state.current_page == 'stored_data':
    show_stored_data_page()
elif st.session_state.current_page == 'dashboard':
    show_dashboard_page()
elif st.session_state.current_page == 'ocr':
    show_ocr_page()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📄 InvoiceIQ: Intelligent Document Processing Platform | Powered by Advanced OCR & AI Technology</p>
</div>
""", unsafe_allow_html=True)

