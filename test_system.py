#!/usr/bin/env python3
"""
Test script for the Stock Price Predictor system
This script tests the core functionality without requiring the web interface
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_predictor import StockPredictor

def test_system():
    """Test the complete stock prediction system"""
    
    print("🧪 Testing Stock Price Predictor System")
    print("=" * 50)
    
    # Test parameters
    ticker = "AAPL"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        print(f"📊 Testing with ticker: {ticker}")
        print(f"📅 Date range: {start_date} to {end_date}")
        print()
        
        # Initialize predictor
        print("1️⃣ Initializing StockPredictor...")
        predictor = StockPredictor(ticker, start_date, end_date)
        print("✅ Initialization successful")
        print()
        
        # Test data collection
        print("2️⃣ Testing data collection...")
        data = predictor.collect_data()
        print(f"✅ Collected {len(data)} days of data")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
        print()
        
        # Test feature engineering
        print("3️⃣ Testing feature engineering...")
        data_with_features = predictor.add_technical_indicators()
        print(f"✅ Added technical indicators")
        print(f"   Final data shape: {data_with_features.shape}")
        print(f"   Features: {[col for col in data_with_features.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]}")
        print()
        
        # Test data preparation
        print("4️⃣ Testing data preparation...")
        X_train, X_test, y_train, y_test, features = predictor.prepare_data()
        print(f"✅ Data prepared successfully")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Features used: {len(features)}")
        print()
        
        # Test model training (with reduced epochs for testing)
        print("5️⃣ Testing model training...")
        print("   ⚠️  Using reduced epochs (10) for testing...")
        history = predictor.train_model(epochs=10, batch_size=32)
        print("✅ Model training completed")
        print()
        
        # Test predictions
        print("6️⃣ Testing predictions...")
        predictions = predictor.predict_future(7)  # 7 days prediction
        print(f"✅ Generated predictions for {len(predictions)} days")
        print(f"   Prediction range: {predictions['Date'].min()} to {predictions['Date'].max()}")
        print(f"   Price range: ${predictions['Close'].min():.2f} - ${predictions['Close'].max():.2f}")
        print()
        
        # Test model evaluation
        print("7️⃣ Testing model evaluation...")
        metrics = predictor.evaluate_model()
        print("✅ Model evaluation completed")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   MAE: {metrics['MAE']:.4f}")
        print(f"   R²: {metrics['R2']:.4f}")
        print()
        
        print("🎉 All tests passed successfully!")
        print("=" * 50)
        print("✅ The stock prediction system is working correctly")
        print("🚀 You can now run 'python app.py' to start the web interface")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print("🔧 Please check your installation and dependencies")
        return False

def test_quick_prediction():
    """Quick test with minimal data"""
    
    print("\n🔬 Quick Prediction Test")
    print("=" * 30)
    
    try:
        # Use a shorter date range for quick testing
        ticker = "AAPL"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"📊 Quick test with {ticker} (6 months of data)")
        
        predictor = StockPredictor(ticker, start_date, end_date)
        predictor.collect_data()
        predictor.add_technical_indicators()
        
        # Quick training with minimal epochs
        history = predictor.train_model(epochs=5, batch_size=16)
        
        # Quick prediction
        predictions = predictor.predict_future(3)
        
        print(f"✅ Quick test successful!")
        print(f"   Predicted prices for next 3 days:")
        for _, row in predictions.iterrows():
            print(f"   {row['Date']}: ${row['Close']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Stock Price Predictor - System Test")
    print("=" * 40)
    
    # Run comprehensive test
    success = test_system()
    
    if success:
        # Run quick test
        test_quick_prediction()
    
    print("\n📋 Test Summary:")
    if success:
        print("✅ System is ready to use!")
        print("🌐 Run 'python app.py' to start the web interface")
    else:
        print("❌ System needs attention")
        print("🔧 Check the error messages above and fix any issues") 