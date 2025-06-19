# Stock Price Predictor - LSTM Neural Network

A comprehensive stock price prediction system built with Python, Flask, and LSTM (Long Short-Term Memory) neural networks. This application uses advanced machine learning techniques to forecast stock prices with interactive visualizations.

## Features

- **Data Collection**: Automated stock data collection using yfinance
- **Feature Engineering**: Advanced technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **LSTM Model**: Deep learning model for time series prediction
- **Interactive Web Interface**: Modern, responsive Flask web application
- **Real-time Predictions**: Forecast stock prices for 7-60 days
- **Performance Metrics**: RMSE and MAE evaluation
- **Export Options**: Download predictions as CSV or charts as PNG

## Technical Indicators Included

- Moving Averages (5, 20, 50-day)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Stochastic Oscillator
- Average True Range (ATR)
- Price changes and volume indicators

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-seer-web-forecast
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Flask application**
   ```bash
   python app.py
   ```

2. **Open your web browser**
   Navigate to `http://localhost:5000`

3. **Configure the model**
   - Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
   - Select start and end dates for training data
   - Choose prediction window (7-60 days)

4. **Train the model**
   Click "Train Model" and wait for the LSTM to complete training

5. **View results**
   - Interactive price prediction chart
   - Performance metrics (RMSE, MAE)
   - Predicted prices table
   - Download options for CSV and PNG

## Project Structure

```
stock-seer-web-forecast/
├── app.py                 # Main Flask application
├── stock_predictor.py     # LSTM model and data processing
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
└── README.md             # This file
```

## Model Architecture

The LSTM model consists of:
- **Input Layer**: 60 time steps with multiple features
- **LSTM Layers**: 3 LSTM layers with 50 units each
- **Dropout Layers**: 20% dropout for regularization
- **Dense Layers**: 25 units + 1 output unit
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Mean Squared Error

## Data Preprocessing

1. **Data Collection**: Historical stock data from Yahoo Finance
2. **Feature Engineering**: 20+ technical indicators
3. **Normalization**: MinMaxScaler (0-1 range)
4. **Sequence Creation**: 60-day lookback window
5. **Train/Test Split**: 80/20 split for validation

## API Endpoints

- `GET /`: Main web interface
- `POST /train`: Train the LSTM model
- `GET /download_csv`: Download predictions as CSV
- `GET /download_plot`: Download chart as PNG

## Example Usage

```python
from stock_predictor import StockPredictor

# Initialize predictor
predictor = StockPredictor('AAPL', '2022-01-01', '2023-12-31')

# Train model
history = predictor.train_model()

# Make predictions
predictions = predictor.predict_future(30)

# Evaluate model
metrics = predictor.evaluate_model()
print(f"RMSE: {metrics['RMSE']:.2f}")
print(f"MAE: {metrics['MAE']:.2f}")
```

## Performance Considerations

- **Training Time**: 2-5 minutes depending on data size
- **Memory Usage**: ~2-4GB RAM for typical datasets
- **GPU Support**: Automatically uses GPU if available
- **Data Requirements**: Minimum 200 days of historical data recommended

## Troubleshooting

### Common Issues

1. **"No data found for ticker"**
   - Verify the ticker symbol is correct
   - Check if the stock is publicly traded
   - Ensure the date range is valid

2. **Memory errors during training**
   - Reduce the date range
   - Close other applications
   - Use a machine with more RAM

3. **Slow training**
   - Reduce the number of epochs
   - Use a smaller date range
   - Enable GPU acceleration if available

### Dependencies Issues

If you encounter dependency conflicts:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and research purposes only. Stock price predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial advisors and conduct thorough research before making investment decisions.

## Future Enhancements

- [ ] Support for multiple stocks simultaneously
- [ ] Additional ML models (GRU, Transformer)
- [ ] Real-time data streaming
- [ ] Portfolio optimization features
- [ ] Sentiment analysis integration
- [ ] Advanced visualization options
