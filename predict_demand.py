import joblib
import pandas as pd

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("demand_prediction_model.pkl")
features = joblib.load("model_features.pkl")

# -----------------------------
# Encoding Maps
# -----------------------------
season_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Autumn": 3}
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4,
    "Saturday": 5, "Sunday": 6
}

# -----------------------------
# Example Input (EDIT VALUES)
# -----------------------------
price = 300
discount_pct = 10
competitor_price = 320
inventory_level = 200
season = "Winter"
day = "Saturday"

price_after_discount = price * (1 - discount_pct / 100)
price_gap = price - competitor_price
is_weekend = 1 if day in ["Saturday", "Sunday"] else 0

input_data = pd.DataFrame([{
    "price": price,
    "discount_pct": discount_pct,
    "competitor_price": competitor_price,
    "inventory_level": inventory_level,
    "season_encoded": season_map[season],
    "day_encoded": day_map[day],
    "price_after_discount": price_after_discount,
    "price_gap": price_gap,
    "is_weekend": is_weekend
}])

input_data = input_data[features]

# -----------------------------
# Prediction
# -----------------------------
predicted_units = max(0, model.predict(input_data)[0])

print("📦 Predicted Units Sold:", int(predicted_units))
