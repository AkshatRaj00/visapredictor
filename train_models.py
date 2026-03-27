import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

np.random.seed(42)

countries = [
    'Afghanistan','Albania','Algeria','Argentina','Australia','Austria',
    'Bangladesh','Belgium','Brazil','Canada','Chile','China','Colombia',
    'Croatia','Czech Republic','Denmark','Egypt','Ethiopia','Finland',
    'France','Germany','Ghana','Greece','Hungary','India','Indonesia',
    'Iran','Iraq','Ireland','Israel','Italy','Japan','Jordan','Kenya',
    'Malaysia','Mexico','Morocco','Nepal','Netherlands','New Zealand',
    'Nigeria','Norway','Pakistan','Peru','Philippines','Poland','Portugal',
    'Romania','Russia','Saudi Arabia','Serbia','Singapore','South Africa',
    'South Korea','Spain','Sri Lanka','Sweden','Switzerland','Thailand',
    'Turkey','UAE','UK','Ukraine','USA','Vietnam','Zimbabwe'
]

visa_types = ['Student', 'Work', 'Tourist', 'Business', 'Transit', 'Medical', 'Family']

# Realistic processing time rules
base_times = {
    'Tourist': 10, 'Transit': 5, 'Student': 30,
    'Work': 45, 'Business': 20, 'Medical': 15, 'Family': 35
}

country_multipliers = {
    'USA': 1.8, 'UK': 1.6, 'Canada': 1.7, 'Australia': 1.5,
    'Germany': 1.4, 'France': 1.3, 'Japan': 1.2, 'India': 0.9,
    'China': 1.1, 'UAE': 0.8, 'Singapore': 0.7, 'Brazil': 1.3,
    'Russia': 1.5, 'Pakistan': 1.1, 'Bangladesh': 1.0, 'Nigeria': 1.2
}

peak_months = [6, 7, 8, 12, 1]  # Summer + December/January

rows = []
for _ in range(5000):
    country = np.random.choice(countries)
    visa = np.random.choice(visa_types)
    month = np.random.randint(1, 13)

    base = base_times[visa]
    multiplier = country_multipliers.get(country, 1.0)
    peak_factor = 1.3 if month in peak_months else 1.0
    noise = np.random.normal(0, 3)

    days = max(3, round(base * multiplier * peak_factor + noise, 1))
    rows.append({'country': country, 'visa_type': visa, 'month': month, 'processing_days': days})

df = pd.DataFrame(rows)

country_encoder = LabelEncoder()
visa_encoder = LabelEncoder()
df['country_enc'] = country_encoder.fit_transform(df['country'])
df['visa_type_enc'] = visa_encoder.fit_transform(df['visa_type'])

X = df[['country_enc', 'visa_type_enc', 'month']]
y = df['processing_days']

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X, y)

joblib.dump(model, 'rf_model.pkl')
joblib.dump(country_encoder, 'country_encoder.pkl')
joblib.dump(visa_encoder, 'visa_encoder.pkl')

print(f"Model trained on {len(df)} samples")
print(f"Countries: {len(countries)}")
print(f"Visa types: {visa_types}")
print("All model files saved!")
