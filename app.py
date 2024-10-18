import nltk
import re
import streamlit as st
import pickle
import numpy as np
import base64
import PyPDF2  # Added PyPDF2 for better PDF handling

nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained model and TF-IDF vectorizer
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
    """Extracts text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text


def webApp():
    st.set_page_config(page_title="CareerMatch", layout="wide")

    # CSS for background image, chat positioning, and layout improvements
    st.markdown(
        """
        <style>
            body {
                background-image: url('logo.png');
                background-size: cover;
            }
            .stApp {
                background-color: rgba(255, 255, 255, 0.8);
            }
            .upload-section {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 40vh;
            }
            iframe {
                position: fixed;
                top: 100px;
                right: 0;
                height: 700px;
                width: 400px;
                border: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 style="font-size: 40px; text-align: center;">Find the role based on your resume...</h1>', unsafe_allow_html=True)

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

            # Job category mapping
            categoryMapping = {
                15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
                20: "Python Developer", 24: "Web Designing", 12: "HR",
                13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
                18: "Operations Manager", 6: "Data Science", 22: "Sales",
                16: "Mechanical Engineer", 1: "Arts", 7: "Database",
                11: "Electrical Engineering", 14: "Health and Fitness",
                19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
                2: "Automation Testing", 17: "Network Security Engineer",
                21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",
            }

            # Get top 5 predicted categories
            probabilities = model.predict_proba(features)
            topFive = np.argsort(probabilities, axis=1)[:, -5:]
            topFiveLabels = [[categoryMapping[idx] for idx in indices_row] for indices_row in topFive]
            predicted_categories = topFiveLabels[0]

            # Display the top predicted roles
            st.markdown("<h2 style='font-weight: bold; font-size: 24px;'>Based on your resume you can apply to the following roles:</h2>", unsafe_allow_html=True)

            colors = ["#1b2838", "#2a475e", "#3a6b87", "#4a8fb0", "#5db5da"]
            for i, cat in enumerate(predicted_categories):
                st.markdown(f"""
                    <div style="background-color: {colors[i]}; padding: 10px; border-radius: 5px; color: white; margin: 5px 0;">
                        {i + 1} - {cat}
                    </div>
                    """, unsafe_allow_html=True)

    # Adding the chatbot iframe to the right side
    st.markdown(
        """
        <iframe src="https://www.chatbase.co/chatbot-iframe/8M1YwJbV8wk47A0QUA0_T" width="100%" style="height: 100%; min-height: 700px" frameborder="0"></iframe>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    webApp()
