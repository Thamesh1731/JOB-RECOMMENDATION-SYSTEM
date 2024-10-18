import nltk
import re
import streamlit as st
import pickle
import numpy as np
import base64
import PyPDF2  # For handling PDF uploads
from PIL import Image  # Optional: Add logo or images for UI

nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(text):
    """Clean and preprocess resume text."""
    cleanText = re.sub('http\S+\s', ' ', text)
    cleanText = re.sub('#[^\s]+|@[^\s]+', ' ', cleanText)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Job category mapping with corresponding application websites
categoryMapping = {
    15: ("Java Developer", "https://www.indeed.com/q-Java-Developer-jobs.html"),
    23: ("Testing", "https://www.indeed.com/q-Testing-jobs.html"),
    8: ("DevOps Engineer", "https://www.indeed.com/q-DevOps-Engineer-jobs.html"),
    20: ("Python Developer", "https://www.indeed.com/q-Python-Developer-jobs.html"),
    24: ("Web Designing", "https://www.indeed.com/q-Web-Designer-jobs.html"),
    12: ("HR", "https://www.indeed.com/q-HR-jobs.html"),
    13: ("Hadoop", "https://www.indeed.com/q-Hadoop-jobs.html"),
    3: ("Blockchain", "https://www.indeed.com/q-Blockchain-jobs.html"),
    10: ("ETL Developer", "https://www.indeed.com/q-ETL-Developer-jobs.html"),
    18: ("Operations Manager", "https://www.indeed.com/q-Operations-Manager-jobs.html"),
    6: ("Data Science", "https://www.indeed.com/q-Data-Scientist-jobs.html"),
    22: ("Sales", "https://www.indeed.com/q-Sales-jobs.html"),
    16: ("Mechanical Engineer", "https://www.indeed.com/q-Mechanical-Engineer-jobs.html"),
    1: ("Arts", "https://www.indeed.com/q-Arts-jobs.html"),
    7: ("Database", "https://www.indeed.com/q-Database-jobs.html"),
    11: ("Electrical Engineering", "https://www.indeed.com/q-Electrical-Engineer-jobs.html"),
    14: ("Health and Fitness", "https://www.indeed.com/q-Fitness-jobs.html"),
    19: ("PMO", "https://www.indeed.com/q-PMO-jobs.html"),
    4: ("Business Analyst", "https://www.indeed.com/q-Business-Analyst-jobs.html"),
    9: ("DotNet Developer", "https://www.indeed.com/q-.NET-Developer-jobs.html"),
    2: ("Automation Testing", "https://www.indeed.com/q-Automation-Testing-jobs.html"),
    17: ("Network Security Engineer", "https://www.indeed.com/q-Network-Security-jobs.html"),
    21: ("SAP Developer", "https://www.indeed.com/q-SAP-Developer-jobs.html"),
    5: ("Civil Engineer", "https://www.indeed.com/q-Civil-Engineer-jobs.html"),
    0: ("Advocate", "https://www.indeed.com/q-Advocate-jobs.html"),
}

def webApp():
    # Improved UI Layout
    st.set_page_config(page_title="CareerMatch", layout="wide")
    
    # Optional: Add a logo (replace 'logo.png' with your file)
    # logo = Image.open('logo.png')
    # st.image(logo, width=100)

    st.markdown('<h1 style="font-size: 40px; text-align: center;">Find the role based on your resume</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        file = st.file_uploader('Select file to upload', type=['pdf'])

    if file is not None:
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Reset file pointer after reading for display
        file.seek(0)

        # Extract text from the uploaded PDF
        resText = extract_text_from_pdf(file)

        if resText:
            newResume = cleanResume(resText)
            features = tfidf.transform([newResume])

            # Get top 5 predicted categories
            probabilities = model.predict_proba(features)
            topFive = np.argsort(probabilities, axis=1)[:, -5:]
            topFiveLabels = [[categoryMapping[idx] for idx in indices_row] for indices_row in topFive]
            predicted_categories = topFiveLabels[0]

            # Display the top predicted roles with links to job sites
            st.markdown("<h2 style='font-weight: bold; font-size: 24px;'>Based on your resume, you can apply to the following roles:</h2>", unsafe_allow_html=True)
            
            colors = ["#1b2838", "#2a475e", "#3a6b87", "#4a8fb0", "#5db5da"]
            for i, (cat, link) in enumerate(predicted_categories):
                st.markdown(f"""
                    <div style="background-color: {colors[i]}; padding: 10px; border-radius: 5px; color: white; margin: 5px 0;">
                        {i + 1} - <a href="{link}" target="_blank" style="color: white; text-decoration: none;">{cat}</a>
                    </div>
                    """, unsafe_allow_html=True)

    # Chatbot Integration
    st.markdown("""
    <h2 style="text-align: center;">Ask Career-Related Questions</h2>
    <iframe
        src="https://embed.chatbot.com/8M1YwJbV8wk47A0QUA0_T"
        width="100%"
        height="500px"
        frameborder="0">
    </iframe>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    webApp()
