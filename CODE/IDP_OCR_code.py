import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import tempfile
import json

st.title("🔍 Robust PaddleOCR Text Extraction")

@st.cache_resource
def load_ocr():
    return PaddleOCR(lang='en')

def extract_text_from_result(result):
    """Safely extract text from PaddleOCR result (handles both old and new formats)"""
    texts = []
    confidences = []
    
    if not result:
        return texts, confidences
    
    # Handle new OCRResult format (what you're getting)
    if hasattr(result[0], 'rec_texts') or ('rec_texts' in result[0] if isinstance(result[0], dict) else False):
        try:
            # Access the first page result
            page_result = result[0]
            
            # Convert to dict if it's an object with attributes
            if hasattr(page_result, 'rec_texts'):
                rec_texts = page_result.rec_texts
                rec_scores = page_result.rec_scores
            elif isinstance(page_result, dict) and 'rec_texts' in page_result:
                rec_texts = page_result['rec_texts']
                rec_scores = page_result['rec_scores']
            else:
                return texts, confidences
                
            # Extract text and confidence scores
            for text, confidence in zip(rec_texts, rec_scores):
                texts.append(text)
                confidences.append(confidence)
                
        except Exception as e:
            st.write(f"Error extracting from new format: {e}")
            return texts, confidences
    
    # Handle traditional list format (backward compatibility)
    elif isinstance(result, list):
        for page in result:
            if not page:
                continue
                
            for line in page:
                if not line:
                    continue
                    
                try:
                    # Handle different possible result structures
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        # Structure: [coordinates, (text, confidence)]
                        text_info = line[1]
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = str(text_info[0])
                            confidence = float(text_info[1])
                            texts.append(text)
                            confidences.append(confidence)
                        
                        # Alternative structure: [coordinates, text, confidence]
                        elif len(line) >= 3:
                            text = str(line[1])
                            confidence = float(line[2])
                            texts.append(text)
                            confidences.append(confidence)
                            
                except (IndexError, TypeError, ValueError) as e:
                    st.write(f"Skipping line due to error: {e}")
                    continue
    
    return texts, confidences

uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg', 'bmp'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Extract Text"):
        with st.spinner("Processing..."):
            try:
                ocr = load_ocr()
                
                # Method 1: Use temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    image.save(tmp.name)
                    result = ocr.ocr(tmp.name)
                
                # Extract text safely
                texts, confidences = extract_text_from_result(result)
                
                if texts:
                    st.success(f"Found {len(texts)} text elements!")
                    
                    # Display results in a nice format
                    st.subheader("Extracted Text")
                    for i, (text, confidence) in enumerate(zip(texts, confidences)):
                        st.write(f"**{i+1}. {text}** (confidence: {confidence:.3f})")
                    
                    # Show summary
                    st.subheader("Summary")
                    st.write(f"Total text elements: {len(texts)}")
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        st.write(f"Average confidence: {avg_confidence:.3f}")
                        st.write(f"Highest confidence: {max(confidences):.3f}")
                        st.write(f"Lowest confidence: {min(confidences):.3f}")
                
                else:
                    st.error("No text detected. Trying alternative approach...")
                    
                    # Debug: show what we actually received
                    st.subheader("Debug - Raw Result Structure")
                    st.write("Result type:", type(result))
                    if result:
                        st.write("First element type:", type(result[0]))
                        if hasattr(result[0], '__dict__'):
                            st.write("Available attributes:", dir(result[0]))
                        elif isinstance(result[0], dict):
                            st.write("Dictionary keys:", list(result[0].keys()))
                        
            except Exception as e:
                st.error(f"Processing error: {e}")