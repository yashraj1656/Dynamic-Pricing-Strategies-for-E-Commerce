# Dynamic-Pricing-Strategies-for-E-Commerce
# 📊 Demand Analytics & Forecasting Platform (Streamlit)

## 🚀 Project Overview

This project is a **Streamlit-based end-to-end analytics application** developed as part of an internship assignment.
The solution demonstrates a complete analytical workflow:

> **Observe → Analysis → Predict → Forecast**

The application transforms raw retail data into actionable business intelligence by combining:

* **Descriptive Analytics (EDA)**
* **Predictive Modeling (Demand / Sales Prediction)**
* **Time Series Forecasting (Seasonality-Aware)**

The goal is to simulate a real-world analytics product used by pricing and sales teams to monitor performance, estimate future demand, and support strategic decision-making.

---

## 🎯 Business Problem

Retail businesses often struggle with:

* Demand uncertainty
* Pricing optimization
* Seasonal demand fluctuations
* Inventory planning

This platform helps answer key business questions:

* How are sales trending over time?
* What factors influence demand?
* How will demand behave in the next 30 days?
* How do pricing and discounts impact revenue and profit?

---

## 🧠 Analytical Framework

### 1️⃣ Observe — Descriptive Analytics (EDA)

Interactive dashboards provide insights into historical performance and demand behavior.

**Key KPIs**

* Total Revenue
* Total Units Sold
* Average Price
* Average Discount %
* Average Inventory Level

**Visual Analytics**

* Date vs Units Sold
* Date vs Revenue
* Season vs Average Units Sold
* Day of Week vs Average Units Sold
* Heatmap (Season × Day of Week)
* Category vs Units Sold
* Price vs Units Sold
* Price vs Competitor Price

---

### 2️⃣ Predict — Demand Prediction (Regression)

Machine learning model predicts demand based on pricing, competition, and seasonal variables.

**Input Features**

* Price
* Discount %
* Competitor Price
* Inventory Level
* Season (Encoded)
* Day of Week (Encoded)

**Outputs**

* Predicted Units Sold
* Expected Revenue
* Expected Profit

**Business Calculations**

```
revenue = price * predicted_units
profit = (price - cost) * predicted_units
```

---

### 3️⃣ Forecast — Time Series Analysis

Demand is aggregated by date and analyzed for trend and seasonality.

**Process**

* Daily demand aggregation
* Trend & seasonality analysis
* Forecast next 30 days using ARIMA (or equivalent model)

**Output**

* Historical demand + forecast visualization
* Future demand estimation for planning

---

## 🧩 App Navigation

The application contains four main pages:

| Page                       | Purpose                            |
| -------------------------- | ---------------------------------- |
| Dashboard Overview         | Exploratory data analysis & KPIs   |
| Demand Prediction          | ML-based demand estimation         |
| Time Series Forecasting    | Future demand forecasting          |
| Pricing & Revenue Insights | Pricing and profitability analysis |

---

## 🎛 Global Filters (Applied Across All Pages)

* Product Category (multi-select)
* Product ID (optional)
* Date Range
* Season
* Day of Week

```python
filtered_df = df[
    (df["category"].isin(selected_categories)) &
    (df["season"].isin(selected_seasons)) &
    (df["day_of_week"].isin(selected_days)) &
    (df["date"].between(start_date, end_date))
]
```

---

## 📂 Dataset Requirements

The dataset includes:

* `date` — time index
* `product_id`, `category`
* `price`, `discount_pct`, `competitor_price`
* `units_sold` (target)
* `revenue`, `cost`
* `inventory_level`
* `season`, `day_of_week`

---

## 🧱 Tech Stack

**Frontend / App**

* Streamlit

**Visualization**

* Plotly

**Machine Learning**

* Scikit-learn (Regression Models)

**Time Series**

* Statsmodels (ARIMA)

**Data Processing**

* Pandas
* NumPy

---

## 📁 Project Structure

```
demand-analytics-streamlit/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv
├── models/
│   └── demand_model.pkl
├── notebooks/
├── images/
└── .gitignore
```

---

## ▶️ How to Run Locally

### 1. Clone Repository

```
git clone https://github.com/your-username/demand-analytics-streamlit.git
cd demand-analytics-streamlit
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Streamlit App

```
streamlit run app.py
```

---

## 📈 Key Insights Generated

The application enables analysis of:

* Price sensitivity across seasons
* Discount effectiveness by weekday
* Revenue trend variations
* Inventory impact on sales
* Seasonal demand patterns

---

## ⚠️ Common Analytics Practices Applied

* Cleaned and validated data pipeline
* Consistent filter logic across dashboards
* Proper date sorting for forecasting
* Separation of aggregated vs raw data
* Feature encoding for seasonal drivers

---

## 🖼 Screenshots

*Add application screenshots here.*

Example:

```
images/dashboard.png
images/prediction.png
images/forecast.png
images/pricing_insights.png
```

---

## 🔮 Future Enhancements

* Auto model selection & tuning
* Real-time API data ingestion
* Demand anomaly detection
* Advanced forecasting (Prophet / LSTM)
* Deployment with CI/CD pipeline

---

## 👨‍💻 Author

**Yashraj Bhosale**
Data Analytics & Machine Learning Enthusiast

---

## ⭐ Project Significance

This project demonstrates the ability to design a **full analytical lifecycle application** combining:

* Business understanding
* Data analysis
* Machine learning
* Forecasting
* Interactive dashboard design

It reflects practical skills aligned with real-world Data Analyst and Data Scientist roles.
