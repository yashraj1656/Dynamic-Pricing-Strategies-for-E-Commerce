import streamlit as st
import pandas as pd
import plotly.express as px
import math

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing



# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Retail Demand Analytics",
    layout="wide"
)

st.title("📊 Retail Demand Analytics Platform")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\NSDC INTERNSHIP\Final Project\synthetic_retail_demand.csv", parse_dates=["date"])
    return df


df = load_data()


@st.cache_resource
def train_demand_model(train_df, feature_cols, model_type):
    clean_df = train_df.dropna(subset=feature_cols + ["units_sold"])
    if clean_df.shape[0] < 20:
        return None, None

    X = clean_df[feature_cols]
    y = clean_df["units_sold"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type.startswith("Gradient"):
        model = GradientBoostingRegressor(random_state=42)
    elif model_type.startswith("Random"):
        model = RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    metrics = {
        "mae": mean_absolute_error(y_val, preds),
        "rmse": math.sqrt(mean_squared_error(y_val, preds)),
        "r2": r2_score(y_val, preds),
    }

    return model, metrics

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard Overview",
        "Demand Prediction",
        "Time Series Forecasting",
        "Pricing & Revenue Insights"
    ]
)


# -----------------------------
# Global Filters (MANDATORY)
# -----------------------------
st.sidebar.header("Global Filters")

selected_categories = st.sidebar.multiselect(
    "Product Category",
    options=df["category"].unique(),
    default=df["category"].unique()
)

selected_products = st.sidebar.multiselect(
    "Product ID",
    options=df["product_id"].unique(),
    default=df["product_id"].unique()
)

date_range = st.sidebar.date_input(
    "Date Range",
    [df["date"].min(), df["date"].max()]
)

selected_seasons = st.sidebar.multiselect(
    "Season",
    options=df["season"].unique(),
    default=df["season"].unique()
)

selected_days = st.sidebar.multiselect(
    "Day of Week",
    options=df["day_of_week"].unique(),
    default=df["day_of_week"].unique()
)

# -----------------------------
# Apply Filters (Single Source)
# -----------------------------
filtered_df = df[
    (df["category"].isin(selected_categories)) &
    (df["product_id"].isin(selected_products)) &
    (df["season"].isin(selected_seasons)) &
    (df["day_of_week"].isin(selected_days)) &
    (df["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
].copy()

# =============================
# Encode Season & Day (GLOBAL & SAFE)
# =============================
season_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Autumn": 3}
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4,
    "Saturday": 5, "Sunday": 6
}

filtered_df["season_encoded"] = filtered_df["season"].map(season_map)
filtered_df["day_encoded"] = filtered_df["day_of_week"].map(day_map)
filtered_df["price_after_discount"] = (
    filtered_df["price"] * (1 - filtered_df["discount_pct"] / 100)
)
filtered_df["price_gap"] = filtered_df["price"] - filtered_df["competitor_price"]
filtered_df["is_weekend"] = filtered_df["day_of_week"].isin(
    ["Saturday", "Sunday"]
).astype(int)


# -----------------------------
# Page Routing
# -----------------------------


if page == "Dashboard Overview":

    st.subheader("📈 Dashboard Overview")

    # -----------------------------
    # KPIs (Filtered)
    # -----------------------------
    total_revenue = filtered_df["revenue"].sum()
    total_units = filtered_df["units_sold"].sum()
    avg_price = filtered_df["price"].mean()
    avg_discount = filtered_df["discount_pct"].mean()
    avg_inventory = filtered_df["inventory_level"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("Units Sold", f"{total_units:,}")
    col3.metric("Avg Price", f"₹{avg_price:,.0f}")
    col4.metric("Avg Discount %", f"{avg_discount:.1f}%")
    col5.metric("Avg Inventory", f"{avg_inventory:.0f}")

    st.divider()

    # -----------------------------
    # Time & Trend
    # -----------------------------
    daily_trend = filtered_df.groupby("date", as_index=False)[
        ["units_sold", "revenue"]
    ].sum()

    fig_units = px.line(
        daily_trend,
        x="date",
        y="units_sold",
        title="Units Sold Over Time"
    )

    fig_revenue = px.line(
        daily_trend,
        x="date",
        y="revenue",
        title="Revenue Over Time"
    )

    st.plotly_chart(fig_units, use_container_width=True)
    st.plotly_chart(fig_revenue, use_container_width=True)

    st.divider()

    # -----------------------------
    # Seasonality & Weekly Patterns
    # -----------------------------
    season_avg = filtered_df.groupby("season", as_index=False)["units_sold"].mean()
    day_avg = filtered_df.groupby("day_of_week", as_index=False)["units_sold"].mean()

    fig_season = px.bar(
        season_avg,
        x="season",
        y="units_sold",
        title="Average Units Sold by Season"
    )

    fig_day = px.bar(
        day_avg,
        x="day_of_week",
        y="units_sold",
        title="Average Units Sold by Day of Week"
    )

    st.plotly_chart(fig_season, use_container_width=True)
    st.plotly_chart(fig_day, use_container_width=True)

    heatmap_df = filtered_df.groupby(
        ["season", "day_of_week"], as_index=False
    )["units_sold"].mean()

    fig_heatmap = px.density_heatmap(
        heatmap_df,
        x="day_of_week",
        y="season",
        z="units_sold",
        title="Demand Heatmap: Day vs Season",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # -----------------------------
    # Product & Pricing Insights
    # -----------------------------
    cat_units = filtered_df.groupby("category", as_index=False)["units_sold"].sum()

    fig_category = px.bar(
        cat_units,
        x="category",
        y="units_sold",
        title="Units Sold by Category"
    )

    fig_price_demand = px.scatter(
        filtered_df,
        x="price",
        y="units_sold",
        color="category",
        title="Price vs Units Sold",
        trendline="ols"
    )

    fig_competition = px.scatter(
        filtered_df,
        x="price",
        y="competitor_price",
        title="Price vs Competitor Price"
    )

    st.plotly_chart(fig_category, use_container_width=True)
    st.plotly_chart(fig_price_demand, use_container_width=True)
    st.plotly_chart(fig_competition, use_container_width=True)

elif page == "Demand Prediction":

    st.subheader("🤖 Demand Prediction")

    st.markdown("""
    This module predicts **units sold** based on pricing, competition,
    inventory availability, and seasonal drivers.
    """)

    # -----------------------------
    # Feature Selection
    # -----------------------------
    features = [
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

    # -----------------------------
    # Model Settings & Training
    # -----------------------------
    model_type = st.selectbox(
        "Model Type",
        [
            "Gradient Boosting (Recommended)",
            "Random Forest",
            "Linear Regression"
        ]
    )

    model, metrics = train_demand_model(filtered_df, features, model_type)
    if model is None:
        st.warning("Not enough data after filters to train a model.")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Validation MAE", f"{metrics['mae']:.1f}")
    m2.metric("Validation RMSE", f"{metrics['rmse']:.1f}")
    m3.metric("Validation R2", f"{metrics['r2']:.3f}")

    st.divider()

    # -----------------------------
    # Scenario Inputs
    # -----------------------------
    st.markdown("### 🔧 Pricing Scenario Inputs")

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Price", 50.0, 1000.0, 300.0)
        discount = st.slider("Discount %", 0, 50, 10)
        inventory = st.number_input("Inventory Level", 10, 1000, 200)

    with col2:
        competitor_price = st.number_input("Competitor Price", 50.0, 1000.0, 320.0)
        season = st.selectbox("Season", list(season_map.keys()))
        day = st.selectbox("Day of Week", list(day_map.keys()))

    price_after_discount = price * (1 - discount / 100)
    price_gap = price - competitor_price
    is_weekend = 1 if day in ["Saturday", "Sunday"] else 0

    input_df = pd.DataFrame([{
        "price": price,
        "discount_pct": discount,
        "competitor_price": competitor_price,
        "inventory_level": inventory,
        "season_encoded": season_map[season],
        "day_encoded": day_map[day],
        "price_after_discount": price_after_discount,
        "price_gap": price_gap,
        "is_weekend": is_weekend
    }])

    # -----------------------------
    # Prediction
    # -----------------------------
    predicted_units = model.predict(input_df)[0]
    predicted_units = max(0, min(predicted_units, inventory))

    avg_cost = filtered_df["cost"].mean()
    expected_revenue = price * predicted_units
    expected_profit = (price - avg_cost) * predicted_units

    st.divider()

    # -----------------------------
    # Outputs
    # -----------------------------
    o1, o2, o3 = st.columns(3)

    o1.metric("Predicted Units Sold", f"{predicted_units:.0f}")
    o2.metric("Expected Revenue", f"₹{expected_revenue:,.0f}")
    o3.metric("Expected Profit", f"₹{expected_profit:,.0f}")

    st.caption(
        "📌 Insight: Demand responds negatively to price increases and positively "
        "to discounts, with seasonal and weekly variations."
    )


elif page == "Time Series Forecasting":

    st.subheader("⏳ Time Series Forecasting")

    # Aggregate demand by date (MANDATORY)
    daily_demand = (
        filtered_df.groupby("date")["units_sold"]
        .sum()
        .sort_index()
        .asfreq("D", fill_value=0)
    )

    st.markdown("### Forecast Model")
    ts_model = st.selectbox(
        "Model",
        ["Holt-Winters (Weekly Seasonality)", "ARIMA (1,1,0)"]
    )

    if ts_model.startswith("Holt-Winters") and len(daily_demand) < 14:
        st.warning("Not enough data for weekly seasonality; falling back to ARIMA.")
        ts_model = "ARIMA (1,1,0)"

    if ts_model.startswith("Holt-Winters"):
        model = ExponentialSmoothing(
            daily_demand,
            trend="add",
            seasonal="add",
            seasonal_periods=7
        )
        results = model.fit()
        forecast = results.forecast(steps=30)
    else:
        model = ARIMA(daily_demand, order=(1, 1, 0))
        results = model.fit()
        forecast = results.forecast(steps=30)

    forecast_df = pd.DataFrame({
        "date": pd.date_range(
            start=daily_demand.index.max() + pd.Timedelta(days=1),
            periods=30,
            freq="D"
        ),
        "forecast_units": forecast
    })

    hist_df = daily_demand.reset_index(name="units_sold")

    st.plotly_chart(
        px.line(hist_df, x="date", y="units_sold", title="Historical Demand"),
        use_container_width=True
    )

    st.plotly_chart(
        px.line(
            forecast_df,
            x="date",
            y="forecast_units",
            title="30-Day Demand Forecast"
        ),
        use_container_width=True
    )


elif page == "Pricing & Revenue Insights":

    st.subheader("💰 Pricing & Revenue Insights")

    st.markdown("""
    This section evaluates how **price, discounts, seasonality, and inventory**
    influence demand and revenue performance.
    """)

    st.divider()

    # -----------------------------
    # Price vs Units Sold (by Season)
    # -----------------------------
    st.plotly_chart(
        px.scatter(
            filtered_df,
            x="price",
            y="units_sold",
            color="season",
            trendline="ols",
            title="Price vs Units Sold (Season-wise)"
        ),
        use_container_width=True
    )

    st.caption(
        "📌 Insight: Higher prices generally reduce demand, with elasticity varying by season."
    )

    st.divider()

    # -----------------------------
    # Discount Effectiveness (by Day)
    # -----------------------------
    st.plotly_chart(
        px.scatter(
            filtered_df,
            x="discount_pct",
            y="units_sold",
            color="day_of_week",
            title="Discount % vs Units Sold (Day-wise)"
        ),
        use_container_width=True
    )

    st.caption(
        "📌 Insight: Discounts are more effective on weekends and high-traffic days."
    )

    st.divider()

    # -----------------------------
    # Revenue Over Time (Season-wise)
    # -----------------------------
    revenue_trend = filtered_df.groupby(
        ["date", "season"], as_index=False
    )["revenue"].sum()

    st.plotly_chart(
        px.line(
            revenue_trend,
            x="date",
            y="revenue",
            color="season",
            title="Revenue Trend Over Time (Season-wise)"
        ),
        use_container_width=True
    )

    st.caption(
        "📌 Insight: Seasonal revenue peaks indicate periods of higher pricing power."
    )

    st.divider()

    # -----------------------------
    # Inventory Impact
    # -----------------------------
    st.plotly_chart(
        px.scatter(
            filtered_df,
            x="inventory_level",
            y="units_sold",
            title="Inventory Level vs Units Sold"
        ),
        use_container_width=True
    )

    st.caption(
        "📌 Insight: Sales are capped by inventory, highlighting supply-side constraints."
    )

