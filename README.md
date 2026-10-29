# Movie Recommendation System Using Machine Learning

Content-based movie recommender built with Python, Scikit-learn and Streamlit.

## Files
- app.py - Streamlit web app
- recommender.py - TF-IDF + cosine similarity logic
- movies.csv - sample dataset (40 movies)
- requirements.txt - dependencies
- Project_Report.txt - project report

## Run (Windows PowerShell)
```
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open http://localhost:8501 in the browser.

## Note
The dataset is a small sample made for demonstration. Results are not real-world recommendations.
