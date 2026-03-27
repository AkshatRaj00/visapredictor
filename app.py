import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime
from predict import predict_processing_time, MODEL_LOADED, LOAD_ERROR
from predict import reload_models
import predict as _predict_module
from train import train_from_dataframe
import google.generativeai as genai

# ── Gemini setup ───────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyDK5y4-vr8f3A4JP2yHNsVi0C8iGrEXvtM"  # apni key yahan paste karo
genai.configure(api_key=GEMINI_API_KEY)

def get_ai_insight(country, visa_type, days, lower, upper):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"""You are a visa processing expert. Give a brief 2-3 sentence practical insight for:
- Country: {country}
- Visa Type: {visa_type}
- Estimated processing time: {days} days (range: {lower} to {upper} days)
Give actionable advice. Be concise and direct. No bullet points."""
        )
        return response.text.strip()
    except Exception:
        return ""

st.set_page_config(
    page_title="Visa Processing Time Estimator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    * { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }
    #MainMenu, footer, header { visibility: hidden; }

    .stApp {
        background: #07080f;
        background-image:
            radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,102,241,0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,185,129,0.08) 0%, transparent 55%);
    }
    .block-container { padding: 0 !important; max-width: 100% !important; }

    .navbar {
        background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.07);
        backdrop-filter: blur(20px); padding: 0 3rem; height: 60px;
        display: flex; align-items: center; justify-content: space-between;
        position: sticky; top: 0; z-index: 999;
    }
    .navbar-brand { color: #fff; font-family: 'Syne', sans-serif !important; font-size: 1rem; font-weight: 700; letter-spacing: -0.2px; }
    .navbar-brand span { color: #818cf8; }
    .navbar-badge {
        background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white;
        font-size: 0.68rem; font-weight: 600; padding: 0.28rem 0.9rem;
        border-radius: 20px; letter-spacing: 0.5px; text-transform: uppercase;
    }
    .page-wrapper { max-width: 1140px; margin: 0 auto; padding: 3rem 2rem; }
    .page-header { margin-bottom: 3rem; }
    .page-header h1 {
        font-family: 'Syne', sans-serif !important; font-size: 2.2rem; font-weight: 800;
        color: #f1f5f9; margin: 0 0 0.5rem 0; letter-spacing: -1px; line-height: 1.1;
    }
    .page-header h1 span { color: #818cf8; }
    .page-header p { font-size: 0.9rem; color: #64748b; margin: 0; line-height: 1.7; max-width: 520px; }

    .status-trained {
        display: inline-flex; align-items: center; gap: 0.45rem;
        background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25);
        color: #34d399; font-size: 0.72rem; font-weight: 600;
        padding: 0.3rem 0.9rem; border-radius: 20px; margin-bottom: 2rem;
    }
    .status-dot { width: 7px; height: 7px; background: #10b981; border-radius: 50%; box-shadow: 0 0 6px #10b981; display: inline-block; }
    .status-untrained {
        display: inline-flex; align-items: center; gap: 0.45rem;
        background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.25);
        color: #fbbf24; font-size: 0.72rem; font-weight: 600;
        padding: 0.3rem 0.9rem; border-radius: 20px; margin-bottom: 2rem;
    }
    .status-dot-warn { width: 7px; height: 7px; background: #f59e0b; border-radius: 50%; box-shadow: 0 0 6px #f59e0b; display: inline-block; }

    .card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 1.8rem; backdrop-filter: blur(10px); }
    .card-header {
        font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.2px; color: #475569; margin-bottom: 1.5rem;
        padding-bottom: 0.9rem; border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    label { font-size: 0.82rem !important; font-weight: 500 !important; color: #94a3b8 !important; }
    .stSelectbox > div > div {
        border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important;
        background: rgba(255,255,255,0.05) !important; color: #f1f5f9 !important; font-size: 0.88rem !important;
    }
    .stDateInput > div > div input {
        border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important;
        background: rgba(255,255,255,0.05) !important; color: #f1f5f9 !important; font-size: 0.88rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; color: white !important;
        border: none !important; border-radius: 10px !important; font-size: 0.88rem !important;
        font-weight: 600 !important; padding: 0.65rem 1.5rem !important; width: 100%;
        box-shadow: 0 4px 20px rgba(79,70,229,0.35);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(79,70,229,0.5) !important; }

    .metrics-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1.4rem 0; }
    .metric-item { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); border-radius: 12px; padding: 1rem 1.2rem; }
    .metric-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #34d399; margin-bottom: 0.35rem; }
    .metric-val { font-family: 'Syne', sans-serif !important; font-size: 1.5rem; font-weight: 700; color: #ecfdf5; }

    .warn-box { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2); border-radius: 12px; padding: 1.2rem 1.5rem; font-size: 0.85rem; color: #fbbf24; line-height: 1.7; }
    .warn-box strong { color: #fde68a; }

    .stFileUploader > div { border: 2px dashed rgba(99,102,241,0.3) !important; border-radius: 12px !important; background: rgba(79,70,229,0.05) !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: rgba(255,255,255,0.04); border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08); padding: 0.3rem; margin-bottom: 2rem; width: fit-content;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 8px !important; padding: 0.45rem 1.6rem !important; font-size: 0.85rem !important; font-weight: 500 !important; color: #475569 !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; color: white !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .footer { text-align: center; color: #334155; font-size: 0.75rem; padding: 2rem 0 1.5rem; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 4rem; }
    .empty-state { min-height: 340px; display: flex; align-items: center; justify-content: center; flex-direction: column; gap: 0.6rem; background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.07); border-radius: 18px; }
    .empty-text { font-size: 0.85rem; color: #334155; }
    .stSpinner p { color: #64748b !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="navbar">
    <span class="navbar-brand">Visa<span>IQ</span></span>
    <span class="navbar-badge">Infosys Milestone 4</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>Processing Time <span>Estimator</span></h1>
    <p>Upload historical visa records to train the AI model. Get intelligent, data-driven estimates powered by machine learning and AI analysis.</p>
</div>
""", unsafe_allow_html=True)

if MODEL_LOADED:
    st.markdown('<div class="status-trained"><span class="status-dot"></span> Model Ready — AI Analysis Enabled</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-untrained"><span class="status-dot-warn"></span> No Model — Upload data to train</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Predict", "Train Model"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if not MODEL_LOADED:
        st.markdown("""
        <div class="warn-box">
            No trained model available. Go to the <strong>Train Model</strong> tab,
            upload your historical visa data CSV, and train the model first.
        </div>
        """, unsafe_allow_html=True)
    else:
        available_countries  = sorted(list(_predict_module.country_encoder.classes_))
        available_visa_types = sorted(list(_predict_module.visa_encoder.classes_))

        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown('<div class="card"><div class="card-header">Application Details</div>', unsafe_allow_html=True)
            with st.form("predict_form"):
                country   = st.selectbox("Country of Application", available_countries)
                visa_type = st.selectbox("Visa Type", available_visa_types)
                app_date  = st.date_input("Application Date", value=datetime.today())
                st.markdown("<br>", unsafe_allow_html=True)
                predict_btn = st.form_submit_button("Get Estimate")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            if predict_btn:
                with st.spinner("Calculating estimate..."):
                    result, error = predict_processing_time({
                        "country": country,
                        "visa_type": visa_type,
                        "application_date": str(app_date)
                    })

                if result:
                    confidence_pct = min(99, max(60, round((1 - result['days'] / 120) * 100)))

                    with st.spinner("Getting AI insight..."):
                        ai_insight = get_ai_insight(
                            country, visa_type,
                            result['days'], result['lower'], result['upper']
                        )

                    ai_section = ""
                    if ai_insight:
                        ai_section = f"""
                        <div style="background:linear-gradient(135deg,rgba(79,70,229,0.12),rgba(124,58,237,0.08));
                             border:1px solid rgba(99,102,241,0.3); border-radius:14px;
                             padding:1.2rem 1.5rem; margin-top:1.2rem;">
                            <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase;
                                 letter-spacing:1px; color:#818cf8; margin-bottom:0.6rem;">AI Insight</div>
                            <div style="font-size:0.85rem; color:#94a3b8; line-height:1.75;">{ai_insight}</div>
                        </div>"""

                    result_html = f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; font-family:'DM Sans',sans-serif; }}
body {{ background:transparent; padding:4px; }}
.rc {{ background:rgba(255,255,255,0.04); border:1px solid rgba(99,102,241,0.25); border-radius:18px; padding:2rem; position:relative; overflow:hidden; }}
.rc::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#4f46e5,#7c3aed,#06b6d4); }}
.rh {{ font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:#475569; margin-bottom:1rem; }}
.rn {{ font-family:'Syne',sans-serif; font-size:5rem; font-weight:800; background:linear-gradient(135deg,#818cf8,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1; letter-spacing:-3px; display:inline-block; }}
.ru {{ font-size:1.1rem; font-weight:500; color:#64748b; margin-left:0.4rem; vertical-align:middle; }}
.dv {{ height:1px; background:rgba(255,255,255,0.06); margin:1.4rem 0; }}
.rr {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1.4rem; }}
.ri {{ background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:1rem 1.2rem; }}
.ril {{ font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#475569; margin-bottom:0.35rem; }}
.riv {{ font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#e2e8f0; }}
.al {{ font-size:0.78rem; color:#64748b; margin-bottom:0.5rem; display:flex; justify-content:space-between; }}
.at {{ height:5px; background:rgba(255,255,255,0.07); border-radius:99px; overflow:hidden; }}
.af {{ height:100%; background:linear-gradient(90deg,#4f46e5,#06b6d4); border-radius:99px; }}
.in {{ background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:10px; padding:1rem 1.2rem; font-size:0.82rem; color:#475569; line-height:1.7; margin-top:1.2rem; }}
.in strong {{ color:#94a3b8; }}
</style>
</head>
<body>
<div class="rc">
    <div class="rh">Estimated Processing Time</div>
    <div style="margin-bottom:1.4rem;display:flex;align-items:baseline;gap:0.3rem;">
        <span class="rn">{result['days']}</span><span class="ru">days</span>
    </div>
    <div class="dv"></div>
    <div class="rr">
        <div class="ri"><div class="ril">Minimum</div><div class="riv">{result['lower']}</div></div>
        <div class="ri"><div class="ril">Maximum</div><div class="riv">{result['upper']}</div></div>
    </div>
    <div class="al">
        <span>Model Confidence</span>
        <span style="font-weight:600;color:#818cf8;">{confidence_pct}%</span>
    </div>
    <div class="at"><div class="af" style="width:{confidence_pct}%"></div></div>
    <div class="in">
        Based on historical records for <strong>{visa_type}</strong> visa
        applications from <strong>{country}</strong>.
        Actual time may vary depending on document completeness and embassy workload.
    </div>
    {ai_section}
</div>
</body>
</html>"""

                    components.html(result_html, height=650, scrolling=False)

                else:
                    st.error(f"Prediction failed: {error}")
            else:
                st.markdown("""
                <div class="empty-state">
                    <div style="font-size:2.5rem;opacity:0.15;">&#9702;</div>
                    <div class="empty-text">Fill in the details and click Get Estimate</div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    col_info, col_upload = st.columns([1, 1], gap="large")

    with col_info:
        st.markdown("""
        <div class="card">
            <div class="card-header">Required CSV Format</div>
            <p style="font-size:0.85rem;color:#475569;margin:0 0 1rem 0;line-height:1.7;">
                Your CSV file must contain exactly these 4 columns.
                Column names are case-insensitive.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.code("""country,visa_type,application_date,decision_date
India,Student,2024-01-10,2024-02-14
USA,Work,2024-03-01,2024-04-20
UK,Tourist,2024-06-15,2024-06-30
Canada,Business,2024-07-01,2024-08-15""", language="csv")

        st.markdown("""
        <div style="font-size:0.82rem;color:#475569;line-height:2;margin-top:0.5rem;">
            <strong style="color:#64748b;">country</strong> &nbsp;— Country of application<br>
            <strong style="color:#64748b;">visa_type</strong> &nbsp;— Category of visa<br>
            <strong style="color:#64748b;">application_date</strong> &nbsp;— Date submitted (YYYY-MM-DD)<br>
            <strong style="color:#64748b;">decision_date</strong> &nbsp;— Date decision received (YYYY-MM-DD)
        </div>
        """, unsafe_allow_html=True)

    with col_upload:
        st.markdown('<div class="card"><div class="card-header">Upload & Train</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Select your CSV file",
            type=["csv"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            try:
                df_preview = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)

                st.markdown(f"""
                <div style="font-size:0.82rem;color:#475569;margin:0.8rem 0 0.5rem;">
                    <strong style="color:#94a3b8;">{len(df_preview)}</strong> rows detected
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(df_preview.head(5), use_container_width=True, height=200)
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("Train Model on This Data"):
                    with st.spinner("Training model..."):
                        df_train = pd.read_csv(uploaded_file)
                        try:
                            metrics = train_from_dataframe(df_train)
                            reload_models()

                            countries_str = ', '.join(metrics['countries'][:8])
                            if len(metrics['countries']) > 8:
                                countries_str += '...'
                            visa_types_str = ', '.join(metrics['visa_types'])

                            st.markdown(f"""
                            <div class="metrics-row">
                                <div class="metric-item">
                                    <div class="metric-label">Rows Used</div>
                                    <div class="metric-val">{metrics['rows_used']}</div>
                                </div>
                                <div class="metric-item">
                                    <div class="metric-label">Mean Error</div>
                                    <div class="metric-val">{metrics['mae']} days</div>
                                </div>
                                <div class="metric-item">
                                    <div class="metric-label">R2 Score</div>
                                    <div class="metric-val">{metrics['r2']}</div>
                                </div>
                            </div>
                            <div style="font-size:0.82rem;color:#475569;line-height:1.9;">
                                <strong style="color:#64748b;">Countries:</strong> {countries_str}<br>
                                <strong style="color:#64748b;">Visa Types:</strong> {visa_types_str}
                            </div>
                            """, unsafe_allow_html=True)

                            st.success("Model trained successfully. Switch to Predict tab.")

                        except ValueError as ve:
                            st.error(f"Data error: {str(ve)}")
                        except Exception as ex:
                            st.error(f"Training failed: {str(ex)}")
            except Exception as e:
                st.error(f"Could not read file: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    VisaIQ &nbsp;&nbsp;|&nbsp;&nbsp;
    Infosys Springboard Internship &nbsp;&nbsp;|&nbsp;&nbsp;
    Milestone 4 &nbsp;&nbsp;|&nbsp;&nbsp; 2026
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)