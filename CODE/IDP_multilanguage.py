import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import tempfile
import json
import os
from groq import Groq

st.title("🔍 Robust PaddleOCR Text Extraction")

# Language selection - Added Hindi
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

selected_lang_name = st.sidebar.selectbox(
    "Select Document Language",
    options=list(language_options.keys()),
    index=0,
    help="Select the language of the document you want to process"
)

selected_lang_code = language_options[selected_lang_name]

@st.cache_resource
def load_ocr(lang_code):
    return PaddleOCR(lang=lang_code, use_angle_cls=True)

def extract_text_from_result(result):
    """Extract text from PaddleOCR result with robust handling for different formats"""
    texts = []
    confidences = []
    
    if not result:
        return texts, confidences
    
    # Debug: Show the structure of the result
    st.sidebar.write("Result type:", type(result))
    if result:
        st.sidebar.write("First element type:", type(result[0]))
    
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
            
            # Check if this is a dictionary format
            elif isinstance(page, dict) and 'rec_texts' in page:
                # Dictionary format
                for i, text in enumerate(page['rec_texts']):
                    if text and 'rec_scores' in page and i < len(page['rec_scores']):
                        texts.append(text)
                        confidences.append(page['rec_scores'][i])
            
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
                            texts.append(text)
                            confidences.append(confidence)
                        
                        # Format: [coordinates, text, confidence]
                        elif len(line) >= 3:
                            text = str(line[1])
                            confidence = float(line[2])
                            texts.append(text)
                            confidences.append(confidence)
    
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        # Fallback: try to extract any text we can find
        try:
            # Convert the entire result to string and look for text patterns
            result_str = str(result)
            # Simple pattern to find text in quotes (common in OCR results)
            import re
            text_matches = re.findall(r"'(.*?)'", result_str)
            texts.extend(text_matches)
            confidences.extend([0.5] * len(text_matches))  # Default confidence
        except:
            pass
    
    return texts, confidences

# UPDATED FUNCTION: Summarize with Llama
def summarize_with_llama(text_content):
    """Summarize extracted text using Llama via Groq API"""
    try:
        # Check if API key is available
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY environment variable not found. Please set it before running the app.")
            return None
            
        # Initialize Groq client with API key from environment variable
        client = Groq(api_key=api_key)
        
        # Prepare the prompt for summarization
        prompt = f"""
        Please provide a concise and informative summary of the following text extracted from a document.
        Focus on the main points, key information, and overall meaning.
        
        Extracted Text:
        {text_content}
        
        Summary:
        """
        
        # Call the Groq API with Llama model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",  # Using Llama model instead of Mistral
            temperature=0.3,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        
        # Extract the summary from the response
        summary = chat_completion.choices[0].message.content
        return summary
        
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        import traceback
        st.write("Full error details:")
        st.code(traceback.format_exc())
        return None

uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg', 'bmp'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Extract Text"):
        with st.spinner("Processing..."):
            try:
                ocr = load_ocr(selected_lang_code)
                
                # Convert image to numpy array
                img_array = np.array(image.convert('RGB'))
                
                # Process image with PaddleOCR
                result = ocr.ocr(img_array)
                
                # Extract text safely
                texts, confidences = extract_text_from_result(result)
                
                if texts:
                    st.success(f"Found {len(texts)} text elements!")
                    
                    # Display original text
                    st.subheader("Extracted Text")
                    for i, (text, confidence) in enumerate(zip(texts, confidences)):
                        st.write(f"{i+1}. {text} (confidence: {confidence:.3f})")
                    
                    # Show text statistics
                    total_chars = sum(len(text) for text in texts)
                    st.write(f"Total characters: {total_chars}")
                    
                    # Show summary
                    st.subheader("Summary")
                    st.write(f"Total text elements: {len(texts)}")
                    st.write(f"Document language: {selected_lang_name}")
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        st.write(f"Average confidence: {avg_confidence:.3f}")
                        st.write(f"Highest confidence: {max(confidences):.3f}")
                        st.write(f"Lowest confidence: {min(confidences):.3f}")

                    # ---------------------------
                    # 📄 JSON Export
                    # ---------------------------
                    json_output = {
                        "metadata": {
                            "language": selected_lang_name,
                            "total_texts": len(texts),
                            "avg_confidence": float(sum(confidences) / len(confidences)) if confidences else None
                        },
                        "results": []
                    }

                    if result:
                        for page in result:
                            # ✅ Case 1: New dict-like format
                            if isinstance(page, dict) and "rec_texts" in page:
                                rec_texts = page.get("rec_texts", [])
                                rec_scores = page.get("rec_scores", [])
                                rec_boxes = page.get("rec_boxes", [])
                                
                                for i, text in enumerate(rec_texts):
                                    json_output["results"].append({
                                        "text": text,
                                        "confidence": float(rec_scores[i]) if i < len(rec_scores) else None,
                                        "coordinates": rec_boxes[i].tolist() if i < len(rec_boxes) else None
                                    })
                            
                            # ✅ Case 2: Classic list format [[coords, (text, conf)]]
                            elif isinstance(page, list):
                                for line in page:
                                    if line is None or not isinstance(line, (list, tuple)):
                                        continue
                                    try:
                                        coords = line[0]
                                        if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                                            text = str(line[1][0])
                                            conf = float(line[1][1])
                                        else:
                                            text = str(line[1])
                                            conf = None
                                        json_output["results"].append({
                                            "text": text,
                                            "confidence": conf,
                                            "coordinates": coords.tolist() if hasattr(coords, "tolist") else coords
                                        })
                                    except:
                                        continue

                    # Convert to JSON string
                    json_str = json.dumps(json_output, indent=2, ensure_ascii=False)

                    # Show preview (first 5 results only)
                    st.subheader("📄 JSON Preview")
                    if json_output["results"]:
                        st.code(json.dumps({
                            "metadata": json_output["metadata"],
                            "results": json_output["results"][:5]
                        }, indent=2, ensure_ascii=False), language="json")
                    else:
                        st.warning("⚠ Still no results parsed. Check OCR raw output with st.write(result).")

                    # Download full JSON
                    st.download_button(
                        label="💾 Download Full JSON",
                        data=json_str,
                        file_name="ocr_result.json",
                        mime="application/json"
                    )
                    
                    # ---------------------------
                    # UPDATED: LLM Summarization Section with Llama
                    # ---------------------------
                    st.subheader("🤖 AI Text Summarization")
                    
                    # Combine all extracted text for summarization
                    combined_text = " ".join(texts)
                    
                    # Show a preview of the text to be summarized
                    with st.expander("View text to be summarized"):
                        st.text(combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text)
                    
                    # Automatically generate summary with Llama
                    with st.spinner("Generating summary with Llama..."):
                        summary = summarize_with_llama(combined_text)
                        
                        if summary:
                            st.success("Summary generated successfully!")
                            st.subheader("AI Summary")
                            st.write(summary)
                            
                            # Add download button for the summary
                            st.download_button(
                                label="📝 Download Summary",
                                data=summary,
                                file_name="document_summary.txt",
                                mime="text/plain"
                            )
                        else:
                            st.error("Failed to generate summary. Please check your API key and try again.")
                
                else:
                    st.error("No text detected. Please try with a different image or language setting.")
                    # Show debug info
                    st.write("Debug - Raw result structure:")
                    st.write(f"Result type: {type(result)}")
                    if result:
                        st.write(f"First element type: {type(result[0])}")
                        if hasattr(result[0], 'dict'):
                            st.write("Available attributes:", dir(result[0]))
                    
            except Exception as e:
                st.error(f"Processing error: {e}")
                import traceback
                st.write("Full error details:")
                st.code(traceback.format_exc())