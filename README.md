# VisaIQ — Visa Processing Time Estimator

A machine learning powered web app that predicts visa processing times based on historical data, with AI-generated insights powered by Google Gemini.

Built for **Infosys Springboard Internship — Milestone 4**

---

## Features

- Upload historical visa data (CSV) to train a custom ML model
- Predict processing time for any country and visa type
- Confidence score and min/max range for each prediction
- AI-powered insights using Google Gemini for actionable advice
- Clean dark UI with real-time results

---

## Tech Stack

- **Frontend** — Streamlit
- **ML Model** — Scikit-learn (Random Forest / Gradient Boosting)
- **AI Insights** — Google Gemini 1.5 Flash
- **Data** — Pandas, NumPy

---

## CSV Format

Your training data must have these 4 columns:

```csv
country,visa_type,application_date,decision_date
India,Student,2024-01-10,2024-02-14
USA,Work,2024-03-01,2024-04-20
UK,Tourist,2024-06-15,2024-06-30
```

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/AkshatRaj00/visapredictor.git
cd visapredictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key in app.py
GEMINI_API_KEY = "your_key_here"

# 5. Run the app
streamlit run app.py
```

---

## Streamlit Cloud Deploy

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo and set main file as `app.py`
4. Add secret in app settings:
```toml
GEMINI_API_KEY = "your_gemini_key"
```

---

## Project Structure

```
visapredictor/
├── app.py              # Main Streamlit app
├── predict.py          # Prediction logic
├── train.py            # Model training
├── requirements.txt    # Dependencies
└── README.md
```

---

Made by Akshat Raj &nbsp;|&nbsp; Infosys Springboard 2026
