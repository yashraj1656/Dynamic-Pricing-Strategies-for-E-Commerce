import pandas as pd
import joblib
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_TYPE = "gradient_boosting"

# -----------------------------
# Load Clean Data
# -----------------------------
df = pd.read_csv(r"synthetic_retail_demand.csv", parse_dates=["date"])


# -----------------------------
# Encode Season & Day
# -----------------------------
season_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Autumn": 3}
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4,
    "Saturday": 5, "Sunday": 6
}

df["season_encoded"] = df["season"].map(season_map)
df["day_encoded"] = df["day_of_week"].map(day_map)
df["price_after_discount"] = df["price"] * (1 - df["discount_pct"] / 100)
df["price_gap"] = df["price"] - df["competitor_price"]
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

# -----------------------------
# Features & Target
# -----------------------------
FEATURES = [
    "price",
    "discount_pct",
    "competitor_price",
    "inventory_level",
    "season_encoded",
    "day_encoded",
    "price_after_discount",
    "price_gap",
    "is_weekend"
]

X = df[FEATURES]
y = df["units_sold"]

# -----------------------------
# Train / Validate Model
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if MODEL_TYPE == "gradient_boosting":
    model = GradientBoostingRegressor(random_state=42)
elif MODEL_TYPE == "random_forest":
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
else:
    model = LinearRegression()

model.fit(X_train, y_train)

preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
rmse = math.sqrt(mean_squared_error(y_val, preds))
r2 = r2_score(y_val, preds)

# -----------------------------
# Save Model & Metadata
# -----------------------------
joblib.dump(model, "demand_prediction_model.pkl")
joblib.dump(FEATURES, "model_features.pkl")

print(" Demand prediction model trained and saved")
print(f" Validation MAE: {mae:.2f}")
print(f" Validation RMSE: {rmse:.2f}")
print(f" Validation R2: {r2:.3f}")
