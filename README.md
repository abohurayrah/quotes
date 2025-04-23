# Quotation Item Extractor

A Streamlit application that uses Google's Gemini API to extract itemized tables from PDF quotations.

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Google AI API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
   You can get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Running the Application

Run the Streamlit app:
```
streamlit run app.py
```

## How to Use

1. Upload a PDF quotation file
2. Click the "Extract Items Table" button
3. The application will extract and display the table of quoted items

If you haven't set the API key in the `.env` file, you can enter it directly in the sidebar of the application. 