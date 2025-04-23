import streamlit as st
import google.generativeai as genai
import pandas as pd
import io
import logging
import traceback
import os
from dotenv import load_dotenv
import base64

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Keep using the specified model

# --- Gemini Helper Function ---
def get_gemini_response(api_key: str, pdf_file_obj, prompt: str):
    """
    Sends the PDF and prompt to the Gemini API and returns the response.

    Args:
        api_key: The Google AI API key.
        pdf_file_obj: The uploaded PDF file object from Streamlit.
        prompt: The instruction prompt for the Gemini model.

    Returns:
        The text response from the Gemini model or None if an error occurs.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)

        st.info("Uploading PDF to Gemini...")
        logging.info("Uploading PDF...")
        
        # Reset read pointer just in case
        pdf_file_obj.seek(0)
        pdf_bytes = pdf_file_obj.read()
        
        # Convert PDF content to base64 for the Gemini API
        mime_type = pdf_file_obj.type
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Create the content parts
        content = [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_pdf
                        }
                    }
                ]
            }
        ]
        
        logging.info(f"PDF prepared for upload: {len(pdf_bytes)} bytes")
        st.info("PDF Prepared. Generating content...")

        # Make the API call
        logging.info("Calling Gemini API...")
        response = model.generate_content(
            content,
            generation_config={"temperature": 0.2},
            stream=False
        )
        
        logging.info("Gemini API call successful.")
        return response.text

    except Exception as e:
        st.error(f"An error occurred while interacting with the Gemini API:")
        st.error(traceback.format_exc()) # Show detailed error in app
        logging.error(f"Gemini API Error: {e}\n{traceback.format_exc()}")
        return None

# --- Parsing Helper Function ---
def parse_markdown_table(markdown_string: str) -> pd.DataFrame | None:
    """
    Parses a Markdown formatted table string into a pandas DataFrame.
    Handles common Markdown table structures.
    """
    if not markdown_string or "---" not in markdown_string:
         st.warning("Did not find a clear Markdown table structure ('---' separator) in the response.")
         logging.warning("Markdown table separator '---' not found in response.")
         return None

    try:
        # Use StringIO to treat the markdown string as a file, makes it easier for pandas
        # Try to clean up potential LLM artifacts like ```markdown ... ```
        cleaned_md = markdown_string.strip().replace('```markdown', '').replace('```', '').strip()

        # Find the header separator line
        lines = cleaned_md.splitlines()
        separator_index = -1
        for i, line in enumerate(lines):
            if all(c in '-|: ' for c in line.strip()) and '---' in line:
                separator_index = i
                break

        if separator_index == -1 or separator_index == 0:
            st.warning("Could not reliably identify the table header and separator.")
            logging.warning("Could not find table header/separator index.")
            return None

        # Extract table content starting from the header
        table_lines = lines[separator_index-1:] # Include header line

        # Reconstruct the relevant part of the markdown table for pandas
        table_md_for_pandas = "\n".join(table_lines)

        # Use pandas read_csv with separator='|' and skipinitialspace=True
        # Wrap in StringIO
        data = io.StringIO(table_md_for_pandas)
        df = pd.read_csv(data, sep='|', skipinitialspace=True)

        # Basic Cleaning Steps for read_csv output with '|'
        df = df.iloc[1:] # Skip the separator line which is now the first row
        df = df.drop(columns=[df.columns[0], df.columns[-1]], errors='ignore') # Drop empty columns from leading/trailing '|'
        df.columns = [col.strip() for col in df.columns] # Strip whitespace from headers
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x) # Strip whitespace from all string cells
        df = df.dropna(axis=1, how='all') # Drop columns that are entirely empty
        df = df.reset_index(drop=True)

        if df.empty:
            st.warning("Parsed table is empty.")
            logging.warning("Parsed DataFrame is empty after cleaning.")
            return None

        return df

    except Exception as e:
        st.error(f"Failed to parse the extracted text into a table.")
        st.error(traceback.format_exc())
        logging.error(f"Error parsing markdown table: {e}\n{traceback.format_exc()}")
        return None

# --- Streamlit App UI ---
st.set_page_config(page_title="Quotation Extractor", layout="wide")
st.title("📄 Quotation Item Extractor using Gemini")
st.markdown("Upload a PDF quotation, and this app will attempt to extract the itemized table using Google's Gemini Pro model.")

# --- API Key Input ---
st.sidebar.header("Configuration")
# First try to get API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("Enter your Google AI API Key:", type="password")
st.sidebar.markdown(
    "🔑 Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)."
    "\n\n⚠️ **Never share your API key.** This app processes it locally in your browser session, but use caution."
)

# --- File Uploader ---
st.header("1. Upload Quotation PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# --- Extraction ---
if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully.")
    logging.info(f"File uploaded: {uploaded_file.name}, size: {uploaded_file.size}, type: {uploaded_file.type}")

    if api_key:
        st.header("2. Extract Table")
        if st.button("✨ Extract Items Table", key="extract_button"):
            # --- Define the Prompt ---
            # Refine this prompt based on the specific structure of your most common quotations
            prompt = """
            Analyze the provided PDF document, which is a price quotation.
            Your primary goal is to identify and extract the main table listing the quoted items.
            This table typically includes columns like: '#', 'REF', 'Proposed Item', 'Description', 'Unit', 'Model', 'Price', 'Qty', 'Amount', or similar variations.

            Please perform the following steps:
            1. Locate the main itemized list/table within the PDF. Ignore headers, footers, introductory text, terms & conditions, and summary sections unless they are part of the main table structure.
            2. Extract all rows from this table accurately. Pay attention to multi-line descriptions within a single cell if applicable.
            3. Preserve the column structure as closely as possible to the original table.
            4. Format the extracted table clearly using Markdown format. Ensure the table includes a header row and a separator line (e.g., | Header 1 | Header 2 | ... |\n|---|---|...|).
            5. Do NOT include any explanatory text before or after the Markdown table in your final output. Just provide the table itself.
            6. If you cannot confidently identify or extract a main item table, respond with the exact text: "NO_TABLE_FOUND".
            """

            with st.spinner("Processing PDF with Gemini... This may take a minute."):
                # Pass the file object directly
                raw_response = get_gemini_response(api_key, uploaded_file, prompt)

            if raw_response:
                st.subheader("Raw Gemini Response (for debugging):")
                st.text_area("Raw Output", raw_response, height=150)
                logging.info("Received raw response from Gemini.")

                if "NO_TABLE_FOUND" in raw_response:
                    st.warning("Gemini indicated that no clear item table was found in the document.")
                    logging.warning("Gemini response indicated NO_TABLE_FOUND.")
                else:
                    st.header("3. Extracted Table")
                    parsed_df = parse_markdown_table(raw_response)

                    if parsed_df is not None:
                        st.success("Table extracted successfully!")
                        st.dataframe(parsed_df, use_container_width=True)
                        logging.info(f"Successfully parsed DataFrame with shape: {parsed_df.shape}")
                        st.markdown("*(You can usually select and copy cells directly from the table above)*")
                    else:
                        st.error("Could not parse the response into a table. Check the raw response above.")
                        logging.error("Failed to parse the raw response into a DataFrame.")
            else:
                # Error message already shown in get_gemini_response
                logging.error("Did not receive a valid response from Gemini.")

    else:
        st.warning("Please enter your Google AI API Key in the sidebar to proceed.")
else:
    st.info("Upload a PDF file to begin.")

st.markdown("---")
st.markdown("Developed with Streamlit and Google Gemini.")