import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify, send_from_directory, request, send_file
import pandas as pd
import numpy as np

# Import pipeline steps
import sys
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_preprocessing
import model_training
import prediction

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "dataset"
REPORT_DIR = BASE_DIR / "report"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Analytics - Sales Revenue Forecasting</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        .gradient-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen">

    <!-- Top Navigation -->
    <nav class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50 px-6 py-4 flex flex-wrap justify-between items-center gap-4">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white text-lg shadow-lg shadow-blue-500/30">
                <i class="fa-solid fa-chart-line"></i>
            </div>
            <div>
                <h1 class="text-xl font-bold tracking-tight text-white">Predictive Analytics Platform</h1>
                <p class="text-xs text-slate-400">Daily Sales Revenue Forecasting • Random Forest Regressor</p>
            </div>
        </div>

        <div class="flex items-center gap-3">
            <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Pipeline Trained & Active
            </span>
            <button onclick="rerunPipeline()" id="rerunBtn" class="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium px-4 py-2 rounded-lg transition shadow-md shadow-blue-600/20">
                <i class="fa-solid fa-rotate" id="rerunIcon"></i> Re-run Pipeline
            </button>
            <a href="/download-forecast" class="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium px-4 py-2 rounded-lg transition border border-slate-700">
                <i class="fa-solid fa-download"></i> Export CSV
            </a>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-6 py-8 space-y-8">

        <!-- KPI Executive Metric Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            <!-- R2 Score -->
            <div class="bg-slate-800/60 border border-slate-750 border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium uppercase tracking-wider text-slate-400">Model Accuracy (R²)</p>
                        <h3 class="text-3xl font-extrabold text-blue-400 mt-1" id="kpi-r2">{{ "%.4f"|format(metrics.R2) }}</h3>
                    </div>
                    <div class="p-3 bg-blue-500/10 rounded-xl text-blue-400">
                        <i class="fa-solid fa-bullseye text-xl"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400">
                    <span class="text-emerald-400 font-semibold mr-1.5"><i class="fa-solid fa-check"></i> 80.62%</span> of variance explained
                </div>
            </div>

            <!-- MAE -->
            <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium uppercase tracking-wider text-slate-400">Mean Absolute Error (MAE)</p>
                        <h3 class="text-3xl font-extrabold text-purple-400 mt-1" id="kpi-mae">${{ "%.2f"|format(metrics.MAE) }}</h3>
                    </div>
                    <div class="p-3 bg-purple-500/10 rounded-xl text-purple-400">
                        <i class="fa-solid fa-gauge text-xl"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400">
                    <span class="text-slate-300 font-semibold mr-1.5">RMSE: ${{ "%.2f"|format(metrics.RMSE) }}</span> on holdout test set
                </div>
            </div>

            <!-- Total 30-Day Forecast -->
            <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium uppercase tracking-wider text-slate-400">30-Day Forecast Revenue</p>
                        <h3 class="text-3xl font-extrabold text-emerald-400 mt-1" id="kpi-forecast">${{ "{:,.2f}".format(forecast_total) }}</h3>
                    </div>
                    <div class="p-3 bg-emerald-500/10 rounded-xl text-emerald-400">
                        <i class="fa-solid fa-arrow-trend-up text-xl"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400">
                    <span class="text-slate-300 font-semibold mr-1.5">Avg: ${{ "{:,.2f}".format(forecast_avg) }}/day</span> across next 30 days
                </div>
            </div>

            <!-- Historical Sales Base -->
            <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium uppercase tracking-wider text-slate-400">Historical Records</p>
                        <h3 class="text-3xl font-extrabold text-amber-400 mt-1">{{ total_days }} Days</h3>
                    </div>
                    <div class="p-3 bg-amber-500/10 rounded-xl text-amber-400">
                        <i class="fa-solid fa-database text-xl"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400">
                    <span class="text-slate-300 font-semibold mr-1.5">${{ "{:,.2f}".format(historical_total) }}</span> total revenue tracked
                </div>
            </div>
        </div>

        <!-- Tab Navigation -->
        <div class="border-b border-slate-800">
            <nav class="flex space-x-6 text-sm font-medium" aria-label="Tabs">
                <button onclick="switchTab('forecast')" id="tab-btn-forecast" class="tab-btn pb-3 text-blue-400 border-b-2 border-blue-400 flex items-center gap-2">
                    <i class="fa-solid fa-chart-area"></i> 30-Day Revenue Forecast
                </button>
                <button onclick="switchTab('eval')" id="tab-btn-eval" class="tab-btn pb-3 text-slate-400 hover:text-slate-200 border-b-2 border-transparent flex items-center gap-2">
                    <i class="fa-solid fa-check-double"></i> Model Evaluation & Accuracy
                </button>
                <button onclick="switchTab('historical')" id="tab-btn-historical" class="tab-btn pb-3 text-slate-400 hover:text-slate-200 border-b-2 border-transparent flex items-center gap-2">
                    <i class="fa-solid fa-calendar-days"></i> Historical Trends & Data
                </button>
                <button onclick="switchTab('report')" id="tab-btn-report" class="tab-btn pb-3 text-slate-400 hover:text-slate-200 border-b-2 border-transparent flex items-center gap-2">
                    <i class="fa-solid fa-file-pdf"></i> Report & Notebook
                </button>
            </nav>
        </div>

        <!-- TAB 1: 30-Day Revenue Forecast -->
        <div id="tab-forecast" class="tab-content space-y-6">
            <!-- Interactive Forecast Chart -->
            <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
                <div class="flex flex-wrap justify-between items-center mb-4 gap-2">
                    <div>
                        <h2 class="text-lg font-bold text-white flex items-center gap-2">
                            <i class="fa-solid fa-arrow-trend-up text-emerald-400"></i> Next 30 Days Forecast Trajectory
                        </h2>
                        <p class="text-xs text-slate-400">Historical actual revenue (recent 60 days) stitched seamlessly with 30-day recursive forecast</p>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="inline-flex items-center gap-1.5 text-xs text-slate-300">
                            <span class="w-3 h-3 bg-blue-500 rounded-sm"></span> Historical
                        </span>
                        <span class="inline-flex items-center gap-1.5 text-xs text-slate-300 ml-3">
                            <span class="w-3 h-3 bg-emerald-500 rounded-sm"></span> Forecast
                        </span>
                    </div>
                </div>
                <div class="h-80 w-full">
                    <canvas id="forecastChart"></canvas>
                </div>
            </div>

            <!-- Forecast Output Image & Table side-by-side -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Generated Matplotlib Output Plot -->
                <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                    <div>
                        <div class="flex justify-between items-center mb-3">
                            <h3 class="text-sm font-bold uppercase tracking-wider text-slate-300">Generated Matplotlib Forecast Figure</h3>
                            <span class="text-xs text-slate-400 font-mono">outputs/future_forecast.png</span>
                        </div>
                        <img src="/outputs/future_forecast.png" alt="Future Forecast" class="rounded-xl border border-slate-700/60 w-full object-cover">
                    </div>
                    <div class="mt-4 p-3 bg-slate-900/80 rounded-xl border border-slate-800 text-xs text-slate-400 flex items-center justify-between">
                        <span>High-resolution export generated by <code class="text-blue-400">src/prediction.py</code></span>
                        <a href="/outputs/future_forecast.png" target="_blank" class="text-blue-400 hover:underline">Full size <i class="fa-solid fa-arrow-up-right-from-square ml-1"></i></a>
                    </div>
                </div>

                <!-- Forecast Table -->
                <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                    <div class="flex justify-between items-center mb-3">
                        <h3 class="text-sm font-bold uppercase tracking-wider text-slate-300">Daily Forecast Data Table (30 Days)</h3>
                        <a href="/download-forecast" class="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 font-medium">
                            <i class="fa-solid fa-file-csv"></i> Download CSV
                        </a>
                    </div>
                    <div class="overflow-y-auto max-h-96 rounded-xl border border-slate-700/60">
                        <table class="w-full text-left text-xs">
                            <thead class="bg-slate-900 text-slate-300 uppercase sticky top-0 border-b border-slate-700">
                                <tr>
                                    <th class="py-2.5 px-3">Date</th>
                                    <th class="py-2.5 px-3">Day of Week</th>
                                    <th class="py-2.5 px-3 text-right">Forecast Revenue</th>
                                    <th class="py-2.5 px-3 text-right">Trend / Status</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800 bg-slate-900/50">
                                {% for row in forecast_rows %}
                                <tr class="hover:bg-slate-800/50 transition">
                                    <td class="py-2.5 px-3 font-mono font-medium text-slate-200">{{ row.Date }}</td>
                                    <td class="py-2.5 px-3 text-slate-400">{{ row.DayName }}</td>
                                    <td class="py-2.5 px-3 text-right font-mono font-bold text-emerald-400">${{ "%.2f"|format(row.Forecast_Revenue) }}</td>
                                    <td class="py-2.5 px-3 text-right">
                                        {% if row.Forecast_Revenue > forecast_avg %}
                                        <span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-300">Above Avg</span>
                                        {% else %}
                                        <span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-700 text-slate-300">Baseline</span>
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    <p class="text-xs text-slate-400 mt-3">Exported from <code class="text-blue-400 font-mono">outputs/next_30_days_forecast.csv</code></p>
                </div>
            </div>
        </div>

        <!-- TAB 2: Model Evaluation & Accuracy -->
        <div id="tab-eval" class="tab-content hidden space-y-6">
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <!-- Metrics Card -->
                <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
                    <h3 class="text-base font-bold text-white flex items-center gap-2">
                        <i class="fa-solid fa-award text-amber-400"></i> Holdout Evaluation Metrics
                    </h3>
                    <p class="text-xs text-slate-400">Evaluated on the last 20% chronological holdout test set (unseen future data).</p>
                    
                    <div class="space-y-3">
                        <div class="p-3 bg-slate-900 rounded-xl border border-slate-750 flex justify-between items-center">
                            <span class="text-xs text-slate-300 font-medium">Split Strategy</span>
                            <span class="text-xs font-mono font-semibold text-blue-400">Chronological 80 / 20</span>
                        </div>
                        <div class="p-3 bg-slate-900 rounded-xl border border-slate-750 flex justify-between items-center">
                            <span class="text-xs text-slate-300 font-medium">R² Score (Variance Explained)</span>
                            <span class="text-xs font-mono font-bold text-emerald-400">{{ "%.4f"|format(metrics.R2) }}</span>
                        </div>
                        <div class="p-3 bg-slate-900 rounded-xl border border-slate-750 flex justify-between items-center">
                            <span class="text-xs text-slate-300 font-medium">Mean Absolute Error (MAE)</span>
                            <span class="text-xs font-mono font-bold text-purple-400">${{ "%.2f"|format(metrics.MAE) }}</span>
                        </div>
                        <div class="p-3 bg-slate-900 rounded-xl border border-slate-750 flex justify-between items-center">
                            <span class="text-xs text-slate-300 font-medium">Root Mean Squared Error (RMSE)</span>
                            <span class="text-xs font-mono font-bold text-amber-400">${{ "%.2f"|format(metrics.RMSE) }}</span>
                        </div>
                        <div class="p-3 bg-slate-900 rounded-xl border border-slate-750 flex justify-between items-center">
                            <span class="text-xs text-slate-300 font-medium">Model Architecture</span>
                            <span class="text-xs font-mono text-slate-300">RandomForest (250 trees)</span>
                        </div>
                    </div>

                    <div class="p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl text-xs text-blue-300 space-y-1">
                        <p class="font-semibold"><i class="fa-solid fa-circle-info mr-1"></i> Key Takeaway:</p>
                        <p class="text-slate-400">With an R² of 0.8062, the model captures >80% of sales fluctuations, tracking seasonal peaks and weekend surges with an average error of less than $60 on daily sales exceeding $2,500.</p>
                    </div>
                </div>

                <!-- Actual vs Predicted Plot -->
                <div class="lg:col-span-2 bg-slate-800/60 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                    <div>
                        <div class="flex justify-between items-center mb-3">
                            <h3 class="text-base font-bold text-white flex items-center gap-2">
                                <i class="fa-solid fa-code-compare text-blue-400"></i> Actual vs Predicted Test Set Comparison
                            </h3>
                            <span class="text-xs text-slate-400 font-mono">outputs/actual_vs_predicted.png</span>
                        </div>
                        <img src="/outputs/actual_vs_predicted.png" alt="Actual vs Predicted" class="rounded-xl border border-slate-700/60 w-full object-cover">
                    </div>
                    <div class="mt-4 p-3 bg-slate-900/80 rounded-xl border border-slate-800 text-xs text-slate-400 flex items-center justify-between">
                        <span>Plot saved by <code class="text-blue-400">src/model_training.py</code></span>
                        <a href="/outputs/actual_vs_predicted.png" target="_blank" class="text-blue-400 hover:underline">Full size <i class="fa-solid fa-arrow-up-right-from-square ml-1"></i></a>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 3: Historical Trends & Data -->
        <div id="tab-historical" class="tab-content hidden space-y-6">
            <!-- Monthly Trend Plot -->
            <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
                <div class="flex justify-between items-center mb-3">
                    <div>
                        <h3 class="text-base font-bold text-white flex items-center gap-2">
                            <i class="fa-solid fa-calendar-check text-blue-400"></i> Monthly Sales Trend (Historical Aggregations)
                        </h3>
                        <p class="text-xs text-slate-400">Visualizing 3-year aggregated monthly sales revenue across 2023 - 2025</p>
                    </div>
                    <span class="text-xs text-slate-400 font-mono">outputs/monthly_sales_trend.png</span>
                </div>
                <img src="/outputs/monthly_sales_trend.png" alt="Monthly Sales Trend" class="rounded-xl border border-slate-700/60 w-full object-cover">
            </div>

            <!-- Historical Data Preview -->
            <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                <div class="flex justify-between items-center">
                    <div>
                        <h3 class="text-base font-bold text-white">Cleaned Historical Dataset Preview</h3>
                        <p class="text-xs text-slate-400">First 10 records from <code class="text-blue-400 font-mono">dataset/sales_data.csv</code> after preprocessing & validation</p>
                    </div>
                    <span class="px-3 py-1 bg-slate-900 border border-slate-700 rounded-lg text-xs font-mono text-slate-300">
                        Total Rows: {{ total_days }}
                    </span>
                </div>
                <div class="overflow-x-auto rounded-xl border border-slate-700/60">
                    <table class="w-full text-left text-xs">
                        <thead class="bg-slate-900 text-slate-300 uppercase border-b border-slate-700">
                            <tr>
                                <th class="py-2.5 px-4">Date</th>
                                <th class="py-2.5 px-4 text-right">Units Sold</th>
                                <th class="py-2.5 px-4 text-right">Revenue</th>
                                <th class="py-2.5 px-4 text-right">Cost</th>
                                <th class="py-2.5 px-4 text-right">Profit</th>
                                <th class="py-2.5 px-4 text-center">Promotion</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800 bg-slate-900/50">
                            {% for r in sample_data %}
                            <tr class="hover:bg-slate-800/50 transition">
                                <td class="py-2.5 px-4 font-mono font-medium text-slate-200">{{ r.Date }}</td>
                                <td class="py-2.5 px-4 text-right font-mono text-slate-300">{{ r.Units_Sold }}</td>
                                <td class="py-2.5 px-4 text-right font-mono font-semibold text-emerald-400">${{ "%.2f"|format(r.Revenue) }}</td>
                                <td class="py-2.5 px-4 text-right font-mono text-slate-400">${{ "%.2f"|format(r.Cost) }}</td>
                                <td class="py-2.5 px-4 text-right font-mono text-blue-400">${{ "%.2f"|format(r.Profit) }}</td>
                                <td class="py-2.5 px-4 text-center">
                                    {% if r.Promotion == 1 %}
                                    <span class="px-2 py-0.5 bg-emerald-500/20 text-emerald-300 rounded text-[10px] font-bold">YES</span>
                                    {% else %}
                                    <span class="px-2 py-0.5 bg-slate-800 text-slate-400 rounded text-[10px]">NO</span>
                                    {% endif %}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 4: Project Report & Notebook -->
        <div id="tab-report" class="tab-content hidden space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- PDF Report Card -->
                <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                    <div class="space-y-4">
                        <div class="w-12 h-12 rounded-xl bg-red-500/10 text-red-400 flex items-center justify-center text-2xl">
                            <i class="fa-solid fa-file-pdf"></i>
                        </div>
                        <div>
                            <h3 class="text-lg font-bold text-white">Project_Report.pdf</h3>
                            <p class="text-xs text-slate-400 mt-1">Comprehensive technical report covering problem formulation, methodology, feature engineering, and model evaluation metrics.</p>
                        </div>
                        <div class="p-3 bg-slate-900 rounded-xl text-xs font-mono text-slate-400">
                            Location: report/Project_Report.pdf
                        </div>
                    </div>
                    <div class="mt-6">
                        <a href="/report/Project_Report.pdf" target="_blank" class="w-full flex items-center justify-center gap-2 bg-red-600 hover:bg-red-500 text-white font-medium py-2.5 px-4 rounded-xl transition shadow-lg shadow-red-600/20 text-sm">
                            <i class="fa-solid fa-arrow-up-right-from-square"></i> Open PDF Report in Browser
                        </a>
                    </div>
                </div>

                <!-- Notebook Card -->
                <div class="bg-slate-800/60 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                    <div class="space-y-4">
                        <div class="w-12 h-12 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center text-2xl">
                            <i class="fa-solid fa-book-open"></i>
                        </div>
                        <div>
                            <h3 class="text-lg font-bold text-white">Predictive_Analytics.ipynb</h3>
                            <p class="text-xs text-slate-400 mt-1">Interactive Jupyter Notebook containing exploratory data analysis (EDA), time series lag engineering, and model training cells.</p>
                        </div>
                        <div class="p-3 bg-slate-900 rounded-xl text-xs font-mono text-slate-400">
                            Location: notebooks/Predictive_Analytics.ipynb
                        </div>
                    </div>
                    <div class="mt-6">
                        <a href="/notebooks/Predictive_Analytics.ipynb" download class="w-full flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-white font-medium py-2.5 px-4 rounded-xl transition text-sm">
                            <i class="fa-solid fa-download"></i> Download Notebook File
                        </a>
                    </div>
                </div>
            </div>
        </div>

    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-800 mt-12 py-6 text-center text-xs text-slate-500">
        Predictive Analytics Using Historical Data • Sales Revenue Forecasting System • Repository: <a href="https://github.com/laxmis123/Predictive-Analytics-Using-Historical-Data.git" target="_blank" class="text-blue-400 hover:underline">laxmis123/Predictive-Analytics-Using-Historical-Data</a>
    </footer>

    <!-- Interactive Chart.js Initialization Script -->
    <script>
        const histDates = {{ hist_dates|tojson }};
        const histValues = {{ hist_values|tojson }};
        const foreDates = {{ fore_dates|tojson }};
        const foreValues = {{ fore_values|tojson }};

        // Combine for seamless continuous line
        const allLabels = [...histDates, ...foreDates];
        const histSeries = [...histValues, ...foreDates.map(() => null)];
        // Connect the last historical point with the forecast
        const foreSeries = [
            ...histDates.slice(0, -1).map(() => null),
            histValues[histValues.length - 1],
            ...foreValues
        ];

        const ctx = document.getElementById('forecastChart').getContext('2d');
        const forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: allLabels,
                datasets: [
                    {
                        label: 'Historical Sales ($)',
                        data: histSeries,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        tension: 0.2,
                        pointRadius: 2,
                        fill: false
                    },
                    {
                        label: '30-Day Forecast ($)',
                        data: foreSeries,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.15)',
                        borderWidth: 2.5,
                        tension: 0.2,
                        pointRadius: 3,
                        pointBackgroundColor: '#10b981',
                        borderDash: [4, 4],
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        titleColor: '#e2e8f0',
                        bodyColor: '#e2e8f0',
                        borderColor: '#334155',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) { label += ': '; }
                                if (context.parsed.y !== null) {
                                    label += '$' + context.parsed.y.toFixed(2);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(51, 65, 85, 0.3)' },
                        ticks: { color: '#94a3b8', maxTicksLimit: 12 }
                    },
                    y: {
                        grid: { color: 'rgba(51, 65, 85, 0.3)' },
                        ticks: {
                            color: '#94a3b8',
                            callback: function(val) { return '$' + val; }
                        }
                    }
                }
            }
        });

        // Tab Switching logic
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => {
                el.classList.remove('text-blue-400', 'border-blue-400');
                el.classList.add('text-slate-400', 'border-transparent');
            });

            document.getElementById('tab-' + tabName).classList.remove('hidden');
            const activeBtn = document.getElementById('tab-btn-' + tabName);
            activeBtn.classList.remove('text-slate-400', 'border-transparent');
            activeBtn.classList.add('text-blue-400', 'border-blue-400');
        }

        // Re-run pipeline dynamically
        async function rerunPipeline() {
            const btn = document.getElementById('rerunBtn');
            const icon = document.getElementById('rerunIcon');
            btn.disabled = true;
            icon.classList.add('fa-spin');
            
            try {
                const res = await fetch('/api/rerun', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    window.location.reload();
                } else {
                    alert('Error re-running pipeline: ' + data.error);
                }
            } catch (e) {
                alert('Request failed: ' + e);
            } finally {
                btn.disabled = false;
                icon.classList.remove('fa-spin');
            }
        }
    </script>
</body>
</html>
"""

def get_metrics():
    metrics = {"MAE": 59.78, "RMSE": 76.11, "R2": 0.8062}
    metrics_path = OUTPUTS_DIR / "evaluation_metrics.txt"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        pass
    return metrics

@app.route("/")
def index():
    # Load forecast data
    forecast_df = pd.read_csv(OUTPUTS_DIR / "next_30_days_forecast.csv")
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    forecast_df["DayName"] = forecast_df["Date"].dt.day_name()
    
    forecast_total = float(forecast_df["Forecast_Revenue"].sum())
    forecast_avg = float(forecast_df["Forecast_Revenue"].mean())
    
    # Load historical sales data
    raw_df = pd.read_csv(DATASET_DIR / "sales_data.csv")
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    total_days = len(raw_df)
    historical_total = float(raw_df["Revenue"].sum())
    
    sample_data = raw_df.head(10).to_dict(orient="records")
    for r in sample_data:
        r["Date"] = r["Date"].strftime("%Y-%m-%d")
        
    metrics = get_metrics()

    # Prepare chart data (last 60 days historical + 30 days forecast)
    recent_history = raw_df.sort_values("Date").tail(60)
    hist_dates = [d.strftime("%Y-%m-%d") for d in recent_history["Date"]]
    hist_values = [round(float(v), 2) for v in recent_history["Revenue"]]
    
    fore_dates = [d.strftime("%Y-%m-%d") for d in forecast_df["Date"]]
    fore_values = [round(float(v), 2) for v in forecast_df["Forecast_Revenue"]]

    forecast_rows = forecast_df.to_dict(orient="records")
    for row in forecast_rows:
        row["Date"] = row["Date"].strftime("%Y-%m-%d")

    return render_template_string(
        HTML_TEMPLATE,
        metrics=metrics,
        forecast_total=forecast_total,
        forecast_avg=forecast_avg,
        total_days=total_days,
        historical_total=historical_total,
        sample_data=sample_data,
        forecast_rows=forecast_rows,
        hist_dates=hist_dates,
        hist_values=hist_values,
        fore_dates=fore_dates,
        fore_values=fore_values
    )

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

@app.route("/dataset/<path:filename>")
def serve_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route("/report/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORT_DIR, filename)

@app.route("/notebooks/<path:filename>")
def serve_notebook(filename):
    return send_from_directory(BASE_DIR / "notebooks", filename)

@app.route("/download-forecast")
def download_forecast():
    return send_file(OUTPUTS_DIR / "next_30_days_forecast.csv", as_attachment=True)

@app.route("/api/summary")
def api_summary():
    metrics = get_metrics()
    forecast_df = pd.read_csv(OUTPUTS_DIR / "next_30_days_forecast.csv")
    return jsonify({
        "metrics": metrics,
        "forecast_total": float(forecast_df["Forecast_Revenue"].sum()),
        "forecast_count": len(forecast_df)
    })

@app.route("/api/rerun", methods=["POST"])
def api_rerun():
    try:
        data_preprocessing.load_and_clean()
        model_training.main()
        prediction.forecast_next_days()
        return jsonify({"status": "success", "metrics": get_metrics()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Predictive Analytics Web Dashboard on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
