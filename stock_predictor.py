import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize the StockPredictor
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data collection (YYYY-MM-DD)
            end_date (str): End date for data collection (YYYY-MM-DD)
        """
        
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.sequence_length = 60  # Number of time steps to look back
        
    def collect_data(self):
        """Collect historical stock data using yfinance"""
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Reset index to make Date a column
            self.data = self.data.reset_index()
            
            # Ensure we have the required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            print(f"Collected {len(self.data)} days of data for {self.ticker}")
            return self.data
            
        except Exception as e:
            raise Exception(f"Error collecting data: {str(e)}")
    
    def add_technical_indicators(self):
        """Add technical indicators as features"""
        if self.data is None:
            raise ValueError("Data not collected yet. Call collect_data() first.")
        
        # Moving Averages
        self.data['MA5'] = ta.trend.sma_indicator(self.data['Close'], window=5)
        self.data['MA20'] = ta.trend.sma_indicator(self.data['Close'], window=20)
        self.data['MA50'] = ta.trend.sma_indicator(self.data['Close'], window=50)
        
        # RSI
        self.data['RSI'] = ta.momentum.rsi(self.data['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        self.data['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(self.data['Close'])
        self.data['BB_Upper'] = bb.bollinger_hband()
        self.data['BB_Lower'] = bb.bollinger_lband()
        self.data['BB_Middle'] = bb.bollinger_mavg()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['Stoch_K'] = stoch.stoch()
        self.data['Stoch_D'] = stoch.stoch_signal()
        
        # Average True Range (ATR)
        self.data['ATR'] = ta.volatility.average_true_range(self.data['High'], self.data['Low'], self.data['Close'])
        
        # Price changes
        self.data['Price_Change'] = self.data['Close'].pct_change()
        self.data['Price_Change_5'] = self.data['Close'].pct_change(periods=5)
        
        # Volume indicators
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        
        # Remove NaN values
        self.data = self.data.dropna()
        
        print(f"Added technical indicators. Data shape: {self.data.shape}")
        return self.data
    
    def prepare_data(self):
        """Prepare data for LSTM model"""
        if self.data is None:
            raise ValueError("Data not prepared yet. Call add_technical_indicators() first.")
        
        # Select features for the model
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'Stoch_K', 'Stoch_D',
            'ATR', 'Price_Change', 'Price_Change_5', 'Volume_MA'
        ]
        
        # Filter columns that exist in the data
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        # Create feature matrix
        features = self.data[available_features].values
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences for LSTM
        X, y = self.create_sequences(scaled_features, self.sequence_length)
        
        # Split into training and testing sets (80-20 split)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        print(f"Data prepared - Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, available_features
    
    def create_sequences(self, data, sequence_length):
        """Create time series sequences for LSTM"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 3])  # Close price is at index 3
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the LSTM model"""
        # Collect and prepare data
        self.collect_data()
        self.add_technical_indicators()
        X_train, X_test, y_train, y_test, features = self.prepare_data()
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Training Loss: {train_loss[0]:.4f}, Training MAE: {train_loss[1]:.4f}")
        print(f"Test Loss: {test_loss[0]:.4f}, Test MAE: {test_loss[1]:.4f}")
        
        return history
    
    def predict_future(self, days=30):
        """Predict future stock prices"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Get the last sequence from the data
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'Stoch_K', 'Stoch_D',
            'ATR', 'Price_Change', 'Price_Change_5', 'Volume_MA'
        ]
        
        available_features = [col for col in feature_columns if col in self.data.columns]
        features = self.data[available_features].values
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Get the last sequence
        last_sequence = scaled_features[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            current_sequence_reshaped = current_sequence.reshape(1, self.sequence_length, len(available_features))
            
            # Predict next value
            next_pred = self.model.predict(current_sequence_reshaped, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence for next prediction
            # Create a new row with predicted values
            new_row = current_sequence[-1].copy()
            new_row[3] = next_pred  # Update close price
            
            # Update other features based on the prediction
            # This is a simplified approach - in practice, you might want more sophisticated feature updates
            new_row[0] = next_pred * 0.99  # Open price
            new_row[1] = next_pred * 1.02  # High price
            new_row[2] = next_pred * 0.98  # Low price
            
            current_sequence = np.vstack([current_sequence[1:], new_row.reshape(1, -1)])
        
        # Inverse transform predictions
        # We need to create a dummy array with the same shape as original features
        dummy_array = np.zeros((len(predictions), len(available_features)))
        dummy_array[:, 3] = predictions  # Put predictions in the close price column
        
        # Inverse transform
        predictions_rescaled = self.scaler.inverse_transform(dummy_array)[:, 3]
        
        # Create prediction dates
        last_date = self.data['Date'].iloc[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'Date': prediction_dates,
            'Close': predictions_rescaled
        })
        
        return predictions_df
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Prepare data
        X_train, X_test, y_train, y_test, features = self.prepare_data()
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        } 