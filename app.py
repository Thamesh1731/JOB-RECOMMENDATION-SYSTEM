import nltk
import re
import streamlit as st
import pickle
import numpy as np


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


def webApp():
    st.set_page_config(page_title="CareerMatch")

    # st.markdown(style, unsafe_allow_html=True)
    # st.title('Discover the role that suits you best...')
    st.markdown('<h1 style="font-size: 40px;">Discover the role that suits you best...</h1>', unsafe_allow_html=True)

    file = st.file_uploader('Select file to upload', type=['pdf'])

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
        st.write("Based on your resume you can apply to the following roles:")
        for i, cat in enumerate(predicted_categories):
            st.write(f'{i + 1} - {cat}')

if __name__ == "__main__":
    webApp()