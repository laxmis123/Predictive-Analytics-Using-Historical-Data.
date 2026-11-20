# Predictive Analytics Using Historical Data
## Sales Revenue Forecasting & Interactive Web Dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Objective
Leverage historical daily sales transaction data to analyze seasonal trends, engineer lag and rolling window features, train a machine learning model, and forecast sales revenue for the next 30 days. Includes a modern browser dashboard to visualize all model outputs, metrics, and projections interactively.

---

### Project Structure

```
├── dataset/
│   └── sales_data.csv                 # Daily sales transaction dataset (2023 - 2025)
├── src/
│   ├── data_preprocessing.py          # Data validation, cleaning, and sanitization
│   ├── model_training.py              # Feature engineering & Random Forest regressor training
│   └── prediction.py                  # Recursive 30-day revenue forecasting
├── models/
│   └── sales_revenue_model.pkl        # Serialized trained Random Forest model
├── outputs/
│   ├── actual_vs_predicted.png        # Evaluation plot: Actual vs Predicted sales on test set
│   ├── monthly_sales_trend.png        # Historical monthly sales revenue aggregation
│   ├── future_forecast.png            # 30-day future sales revenue forecast plot
│   ├── evaluation_metrics.txt         # MAE, RMSE, and R² test metrics
│   └── next_30_days_forecast.csv      # Daily revenue forecast table
├── notebooks/
│   └── Predictive_Analytics.ipynb     # Jupyter Notebook with EDA, training, and plots
├── report/
│   └── Project_Report.pdf             # Comprehensive technical project report
├── app.py                             # Interactive browser dashboard web application
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

### Dataset Details
`dataset/sales_data.csv` contains 1,096 daily transaction records spanning 2023-01-01 through 2025-12-31.
- `Date`: Transaction date (daily)
- `Units_Sold`: Total units sold per day
- `Revenue`: Gross sales revenue
- `Cost`: Estimated operational/product cost
- `Profit`: Net profit (`Revenue - Cost`)
- `Promotion`: Promotional campaign indicator (`1` if active, `0` otherwise)

---

### Quick Start & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/laxmis123/Predictive-Analytics-Using-Historical-Data.git
   cd Predictive-Analytics-Using-Historical-Data
   ```

2. **Set up virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Analytics Pipeline

Run the scripts in sequence from the project root:

1. **Data Preprocessing & Cleaning:**
   ```bash
   python src/data_preprocessing.py
   ```

2. **Model Training & Evaluation:**
   ```bash
   python src/model_training.py
   ```

3. **30-Day Revenue Forecasting:**
   ```bash
   python src/prediction.py
   ```

---

### Viewing Outputs in Browser (Web Dashboard)

Launch the interactive web application to view all outputs, KPI metrics, dynamic charts, and forecast tables directly in your browser:

```bash
python app.py
```

Then open your browser to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

The browser dashboard provides:
- **KPI Overview**: Real-time cards for R² score, MAE, RMSE, historical sales totals, and 30-day forecast projection.
- **Interactive Forecast Explorer**: Dynamic Chart.js trajectory linking historical sales seamlessly with the 30-day forecast.
- **Model Evaluation**: High-resolution actual vs. predicted curves on holdout data.
- **Historical Analysis**: Aggregated monthly revenue trends and sample dataset tables.
- **Report & Notebook Access**: One-click viewing of `Project_Report.pdf` and download of `Predictive_Analytics.ipynb`.
- **One-Click Re-run**: Trigger full model retraining directly from the UI.

---

### Model Performance & Evaluation

The model utilizes a **Random Forest Regressor** trained on engineered date attributes, 1-day and 7-day lags, and 7-day and 30-day rolling averages. The final 20% chronological segment is held out for out-of-sample testing:

| Metric | Holdout Score |
| :--- | :--- |
| **Split Strategy** | Chronological 80 / 20 Split |
| **R² (Coefficient of Determination)** | **0.8062** (Captures >80% of sales variance) |
| **MAE (Mean Absolute Error)** | **$59.78** |
| **RMSE (Root Mean Squared Error)** | **$76.11** |

---

### Outputs Generated
- `outputs/actual_vs_predicted.png`: Out-of-sample performance curve.
- `outputs/monthly_sales_trend.png`: 3-year aggregated monthly revenue chart.
- `outputs/future_forecast.png`: 30-day forecast projection with 90-day history.
- `outputs/next_30_days_forecast.csv`: Tabular daily projections for operational planning.
- `models/sales_revenue_model.pkl`: Serialized model and pipeline state.
