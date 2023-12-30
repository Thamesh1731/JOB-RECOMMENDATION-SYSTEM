import nltk
import re
import streamlit as st
import pickle
import numpy as np
import base64


nltk.download('punkt')
nltk.download('stopwords')

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))



def cleanResume(text):
    cleanText = re.sub('http\S+\s', ' ', text)
    cleanText = re.sub('#[^\s]+|@[^\s]+', ' ', cleanText)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# def show_pdf(file_path):
#     with open(file_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
#     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)
def get_pdf_file_as_base64(file):
    """Converts a file to base64."""
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    return base64_pdf

def webApp():
    st.set_page_config(page_title="CareerMatch")
    st.markdown('<h1 style="font-size: 40px;">Find the role based on your resume...</h1>', unsafe_allow_html=True)

    file = st.file_uploader('Select file to upload', type=['pdf'])

    if file is not None:
        base64_pdf = get_pdf_file_as_base64(file)
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    

    if file is not None:
        try:
            resBytes = file.read()
            resText = resBytes.decode('utf-8')
        except UnicodeDecodeError:
            resText = resBytes.decode('latin-1')

        newResume = cleanResume(resText)
        features = tfidf.transform([newResume])

        categoryMapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        probabilities = model.predict_proba(features)
    
        topFive = np.argsort(probabilities, axis=1)[:, -5:]
        topFiveLabels = [[categoryMapping[idx] for idx in indices_row] for indices_row in topFive]
        predicted_categories = topFiveLabels[0]

        print("Top 5 Predicted Categories:", predicted_categories)
        st.markdown("""
            <h2 style="font-weight: bold; font-size: 24px;">
                Based on your resume you can apply to the following roles:
            </h2>
            """, unsafe_allow_html=True)
        
        colors = ["#1b2838", "#2a475e", "#3a6b87", "#4a8fb0", "#5db5da"]
        for i, cat in enumerate(predicted_categories):
            st.markdown(f"""
                <div style="background-color: {colors[i]}; padding: 10px; border-radius: 5px; color: white; margin: 5px 0;">
                    {i + 1} - {cat}
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    webApp()