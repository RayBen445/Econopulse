# EconoPulse 📈

**Real-Time Economic Indicators & Policy Insight Platform**

![EconoPulse Dashboard](https://github.com/user-attachments/assets/efebf373-925c-4da3-bd38-b56a1f7af1b0)

EconoPulse is an interactive web dashboard for exploring economic indicators (inflation, GDP growth, unemployment, FX rates), comparing countries side-by-side, detecting economic stress signals, and forecasting trends.

---

## Features

| Feature | Description |
|---|---|
| 📊 **Dashboard** | KPI cards + interactive time-series charts for all indicators |
| 🌍 **Country Comparison** | Side-by-side line charts, bar charts, and correlation heatmaps |
| 🚨 **Signal Detection** | Automatic flagging of inflation spikes, GDP contractions, unemployment surges |
| 🔮 **Forecasting** | Holt linear-trend model (falls back to OLS) with configurable horizon |
| 💾 **Export** | Download filtered data and alerts as CSV, or a full text report |
| 🗂 **Data Sources** | Built-in sample data, World Bank Open Data API, or your own CSV upload |

---

## ⭐ Premium Features

Ten advanced analytical features are available in the **⭐ Premium** tab:

| # | Feature | Description |
|---|---|---|
| 1 | 📊 **Economic Health Score** | Composite 0–100 index per country per year. GDP growth and GDP per capita raise the score; inflation and unemployment lower it (weighted, min-max normalised). |
| 2 | 🔮 **Multi-Country Forecast Comparison** | Forecast the same indicator for all selected countries simultaneously on a single chart using Holt's linear-trend model. |
| 3 | 🌡️ **YoY Change Heatmap** | Pivot heatmap of year-over-year absolute changes (percentage points) for any indicator, making acceleration and deceleration immediately visible. |
| 4 | 🔥 **Economic Risk Score Heatmap** | Annual risk score per country = sum of alert severity weights (low=1, medium=2, high=3). Heatmap surface highlights where and when stress concentrated. |
| 5 | 📉 **Recession Detection** | Identifies consecutive years of negative GDP growth (configurable minimum, default 2) and overlays shaded bands on the GDP chart. |
| 6 | 🏆 **Country Percentile Rankings** | Rank selected countries against each other for any indicator and year; bar chart coloured green-to-red by percentile. |
| 7 | 📈 **Moving Average Trend Overlay** | Configurable rolling-mean window (2–10 years) overlaid on the raw series to smooth noise and reveal medium-term trends. |
| 8 | 🎛️ **Custom Alert Thresholds** | Slider-controlled per-indicator thresholds replace the built-in defaults; alerts fire in real time as thresholds are adjusted. |
| 9 | 📊 **Volatility Analysis** | Rolling standard deviation of any indicator for each country; bar chart of average volatility ranks the most erratic economies. |
| 10 | 🔵 **Economic Similarity Clustering** | K-means clustering (k configurable) groups countries by their full indicator profile in a selected year; scatter plot and cluster-assignment table. |

---

## Project Structure

```
Econopulse/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── data/
│   ├── fetcher.py          # Data loading (World Bank API / CSV / sample)
│   └── sample_data.py      # Offline sample data generator
├── analytics/
│   ├── signals.py          # Economic stress signal detection
│   ├── forecasting.py      # Time-series forecasting
│   └── premium.py          # 10 premium analytical functions
├── utils/
│   └── export.py           # CSV & text-report export helpers
└── tests/                  # Pytest test suite (80 tests)
```

---

## Deployment

### Option 1 — Streamlit Community Cloud *(recommended, free)*

The easiest way to share EconoPulse publicly at no cost.

**What you need:**
- A free [Streamlit Community Cloud](https://streamlit.io/cloud) account (sign in with GitHub)
- This repository pushed to a **public** GitHub repo (or a private repo on a paid Streamlit plan)
- No API keys — the World Bank API and the built-in sample data are both public

**Steps:**
1. Go to <https://share.streamlit.io> and click **"New app"**
2. Select your GitHub repository and branch
3. Set **Main file path** to `app.py`
4. Click **Deploy** — Streamlit installs `requirements.txt` automatically

Your app will be live at `https://<your-handle>-econopulse-app-<hash>.streamlit.app`.

---

### Option 2 — Run Locally

**What you need:**
- Python 3.10 or later
- `pip`

```bash
# 1. Clone the repository
git clone https://github.com/RayBen445/Econopulse.git
cd Econopulse

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

### Option 3 — Docker

**What you need:**
- [Docker](https://docs.docker.com/get-docker/) installed

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then build and run:

```bash
docker build -t econopulse .
docker run -p 8501:8501 econopulse
```

Open <http://localhost:8501>.

This Docker image can be pushed to any container registry (Docker Hub, AWS ECR, Google Artifact Registry, Azure Container Registry) and deployed on:
- **AWS** — Elastic Container Service (ECS) / App Runner
- **Google Cloud** — Cloud Run (`gcloud run deploy`)
- **Azure** — Container Apps / App Service
- **Railway / Render / Fly.io** — point to your Docker image or GitHub repo

---

### Option 4 — Heroku

**What you need:**
- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and a free/paid Heroku account

Create a `Procfile` in the project root:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Then deploy:

```bash
heroku create econopulse-app
git push heroku main
heroku open
```

---

## What You Need to Provide (Summary)

| Item | Required? | Notes |
|---|---|---|
| Python 3.10+ | ✅ Yes | All platforms except Streamlit Cloud (managed) |
| `requirements.txt` dependencies | ✅ Yes | Installed automatically on Streamlit Cloud |
| GitHub repository | ✅ For Streamlit Cloud | Public repo for the free tier |
| World Bank API key | ❌ No | The API is open and does not require authentication |
| Database / storage | ❌ No | All data is fetched at runtime or uploaded by the user |
| Environment variables / secrets | ❌ No | No secrets needed for any current feature |
| Custom CSV data | ⚙️ Optional | Upload via the sidebar "Upload CSV" option at runtime |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **Pandas / NumPy** — data manipulation
- **Plotly** — interactive charts
- **statsmodels** — Holt's linear-trend forecasting
- **scikit-learn** — linear regression fallback
- **World Bank Open Data API** — live economic data (optional)

---

## License

This project is licensed under the [MIT License](LICENSE).
