# 🎫 Ticket Price Prophet

A machine learning web application that predicts whether ticket prices will go **UP** 📈 or **DOWN** 📉 based on historical data and market conditions.

## Features

- **Event Selection**: Choose from 10+ different events (concerts, sports, etc.)
- **Price Prediction**: ML model with 88.2% accuracy
- **Interactive Web Interface**: Built with Streamlit
- **Historical Analysis**: Uses price trends from 1, 3, and 7 days ago
- **Visual Charts**: Price trends and probability distributions
- **Real-time Predictions**: Instant results with confidence scores

## Quick Start

### 1. Run the Web App

```bash
# Make script executable (first time only)
chmod +x run_app.sh

# Start the app
./run_app.sh
```

The app will be available at: `http://localhost:8501`

## How It Works

The model uses a **Random Forest Classifier** trained on real ticket price data from multiple events including:

- Coldplay, Lady Gaga, Paul McCartney, Shawn Mendes
- NFL games (Broncos, Chargers, Seahawks)
- US Open Tennis
- And more...

### Key Features Used:

- Current ticket price
- Historical prices (1, 3, 7 days ago)
- Days until event
- Ticket quantity available
- Section and row information
- Event type

### Model Performance:

- **Accuracy**: 88.2%
- **Precision**: 89%
- **Cross-validation**: 85.3% ± 7.5%

## Project Structure

```
ticket-prophet/
├── app.py                          # Streamlit web application
├── train_model.py                  # ML model training script
├── ticket_price_model.joblib       # Trained model (generated)
├── run_app.sh                      # App startup script
├── requirements.txt                # Python dependencies
├── json/                          # Historical ticket data
├── analysis/                      # Jupyter notebooks
│   ├── ticket_env/               # Virtual environment
│   └── *.ipynb                   # Analysis notebooks
└── scripts/                      # CSV datasets
    └── *.csv                     # Processed data files
```

## Data Sources

The model is trained on historical ticket price data from:

- 70+ JSON files with daily price snapshots
- 1,750+ ticket records across 10 events
- Price tracking over 7-day periods
- Multiple venues and seating sections

## Usage Tips

1. **For Best Predictions**: Enter accurate historical price data
2. **New Events**: Model handles unknown events with default encoding
3. **Price Trends**: Look at the historical chart to understand context
4. **Confidence**: Higher probability scores indicate more confident predictions

## Technical Details

- **Framework**: Streamlit for web interface
- **ML Algorithm**: Random Forest Classifier (100 trees, max depth 10)
- **Features**: 12 numerical and categorical features
- **Training Data**: 170 records with balanced classes (45% up, 55% down)
- **Feature Engineering**: Historical price differences and quantity trends

---

**Disclaimer**: Predictions are estimates based on historical data. Actual ticket prices may vary due to market conditions, demand fluctuations, and external factors.