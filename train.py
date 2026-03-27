import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_from_dataframe(df: pd.DataFrame):
    """
    Train model from user-uploaded CSV.
    
    Required columns:
        - country           : string  (e.g. 'India')
        - visa_type         : string  (e.g. 'Student')
        - application_date  : date    (e.g. '2024-03-15')
        - decision_date     : date    (e.g. '2024-04-10')
    
    Returns: metrics dict or raises ValueError
    """

    # ── 1. Validate required columns ────────────────────────────────────────
    required = {'country', 'visa_type', 'application_date', 'decision_date'}
    missing = required - set(df.columns.str.strip().str.lower())
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df.columns = df.columns.str.strip().str.lower()

    # ── 2. Parse dates & compute target ─────────────────────────────────────
    df['application_date'] = pd.to_datetime(df['application_date'], dayfirst=False, errors='coerce')
    df['decision_date']    = pd.to_datetime(df['decision_date'],    dayfirst=False, errors='coerce')
    df = df.dropna(subset=['application_date', 'decision_date'])

    df['processing_days'] = (df['decision_date'] - df['application_date']).dt.days
    df = df[df['processing_days'] > 0]   # remove negative / zero

    if len(df) < 10:
        raise ValueError("Not enough valid rows after cleaning (need at least 10).")

    # ── 3. Feature engineering ───────────────────────────────────────────────
    df['month']     = df['application_date'].dt.month
    df['country']   = df['country'].str.strip().str.title()
    df['visa_type'] = df['visa_type'].str.strip().str.title()

    # ── 4. Encode categoricals ───────────────────────────────────────────────
    country_encoder  = LabelEncoder()
    visa_encoder     = LabelEncoder()
    df['country_enc']   = country_encoder.fit_transform(df['country'])
    df['visa_type_enc'] = visa_encoder.fit_transform(df['visa_type'])

    # ── 5. Train / test split ────────────────────────────────────────────────
    X = df[['country_enc', 'visa_type_enc', 'month']]
    y = df['processing_days']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 6. Train model ───────────────────────────────────────────────────────
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ── 7. Evaluate ──────────────────────────────────────────────────────────
    preds = model.predict(X_test)
    mae   = round(mean_absolute_error(y_test, preds), 2)
    r2    = round(r2_score(y_test, preds), 4)

    # ── 8. Save artifacts ────────────────────────────────────────────────────
    joblib.dump(model,           os.path.join(BASE_DIR, 'rf_model.pkl'))
    joblib.dump(country_encoder, os.path.join(BASE_DIR, 'country_encoder.pkl'))
    joblib.dump(visa_encoder,    os.path.join(BASE_DIR, 'visa_encoder.pkl'))

    return {
        "rows_used":  len(df),
        "mae":        mae,
        "r2":         r2,
        "countries":  sorted(list(country_encoder.classes_)),
        "visa_types": sorted(list(visa_encoder.classes_)),
    }
