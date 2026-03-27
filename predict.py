import pandas as pd
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_LOADED = False
LOAD_ERROR   = None
model = country_encoder = visa_encoder = None


def _load_models():
    global model, country_encoder, visa_encoder, MODEL_LOADED, LOAD_ERROR
    try:
        model           = joblib.load(os.path.join(BASE_DIR, 'rf_model.pkl'))
        country_encoder = joblib.load(os.path.join(BASE_DIR, 'country_encoder.pkl'))
        visa_encoder    = joblib.load(os.path.join(BASE_DIR, 'visa_encoder.pkl'))
        MODEL_LOADED    = True
        LOAD_ERROR      = None
        logger.info("Models loaded successfully")
    except Exception as e:
        MODEL_LOADED = False
        LOAD_ERROR   = str(e)
        logger.error(f"Model load error: {LOAD_ERROR}")


_load_models()


def reload_models():
    """Call this after retraining to hot-reload without restart."""
    _load_models()


def safe_encode(encoder, value):
    val = str(value).strip().title()
    if val in encoder.classes_:
        return int(encoder.transform([val])[0])
    logger.warning(f"Unknown value '{val}', using fallback")
    return int(encoder.transform([encoder.classes_[0]])[0])


def predict_processing_time(data_dict):
    if not MODEL_LOADED:
        return None, f"Model not ready: {LOAD_ERROR}"
    try:
        df    = pd.DataFrame([data_dict])
        month = pd.to_datetime(df['application_date']).iloc[0].month

        country_enc = safe_encode(country_encoder, df['country'].iloc[0])
        visa_enc    = safe_encode(visa_encoder,    df['visa_type'].iloc[0])

        X = pd.DataFrame(
            [[country_enc, visa_enc, month]],
            columns=['country_enc', 'visa_type_enc', 'month']
        )

        prediction = float(model.predict(X)[0])
        prediction = max(1.0, prediction)

        margin = round(prediction * 0.15, 1)
        lower  = max(1, round(prediction - margin, 1))
        upper  = round(prediction + margin, 1)

        return {"days": round(prediction, 1), "lower": lower, "upper": upper}, None

    except Exception as e:
        return None, str(e)
