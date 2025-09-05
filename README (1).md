# CareerMatch – Resume‑Based Job Recommendation System

CareerMatch is a simple web app I built to help job seekers see which roles best match their resume. Upload a PDF, the app cleans and analyzes the text, then suggests the top 5 job categories along with quick links to apply. It’s lightweight, fast, and runs entirely in your browser with Streamlit.

---

## What It Does
- Lets you **upload and preview** your resume (PDF)
- Cleans and processes text using **NLTK**
- Uses a **TF‑IDF vectorizer + machine learning model** to predict job roles
- Shows the **top five categories** with direct **Indeed links**
- Includes an embedded **chatbot** for career questions

---

## How I Built It
This project runs on:
- **Python 3**  
- **Streamlit** for the web UI  
- **scikit‑learn** & **pickle** for the ML model  
- **NLTK** for text cleaning  
- **PyPDF2** for PDF extraction  
- **NumPy** for handling predictions

---

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/career-match.git
   cd career-match
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Add the trained model**
   Place `model.pkl` and `tfidf.pkl` in the project folder.

5. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open the URL shown in your terminal and start exploring roles.

---

## A Few Notes
- Keep your trained model files private if they contain sensitive data.
- The job mapping currently points to **Indeed**; you can edit links for other boards.
- Resume text is processed locally — no external storage or upload.

---

## Future Ideas
- Word/Docx support
- Smarter skill‑based recommendations
- Dashboard of saved searches and tracked roles
