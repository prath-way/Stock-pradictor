from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import io
import base64
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from stock_predictor import StockPredictor
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variable to store the predictor instance
predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    global predictor
    try:
        data = request.get_json()
        ticker1 = data['ticker'].upper()
        ticker2 = data['ticker2'].upper()
        start_date = data['start_date']
        end_date = data['end_date']
        prediction_days = int(data['prediction_days'])

        # First stock
        predictor1 = StockPredictor(ticker1, start_date, end_date)
        history1 = predictor1.train_model()
        predictions1 = predictor1.predict_future(prediction_days)
        plot_data1 = create_interactive_plot(predictor1.data, predictions1, ticker1)
        metrics1 = calculate_metrics(predictor1.data, predictions1)

        # Second stock
        predictor2 = StockPredictor(ticker2, start_date, end_date)
        history2 = predictor2.train_model()
        predictions2 = predictor2.predict_future(prediction_days)
        plot_data2 = create_interactive_plot(predictor2.data, predictions2, ticker2)
        metrics2 = calculate_metrics(predictor2.data, predictions2)

        return jsonify({
            'success': True,
            'plot1': plot_data1,
            'metrics1': metrics1,
            'predictions1': predictions1.to_dict('records'),
            'plot2': plot_data2,
            'metrics2': metrics2,
            'predictions2': predictions2.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/download_csv')
def download_csv():
    global predictor
    if predictor is None:
        return jsonify({'error': 'No model trained yet'}), 400
    try:
        predictions = predictor.predict_future(30)
        historical_data = predictor.data[['Date', 'Close']].copy()
        historical_data['Type'] = 'Historical'
        pred_data = predictions[['Date', 'Close']].copy()
        pred_data['Type'] = 'Predicted'
        combined_data = pd.concat([historical_data, pred_data], ignore_index=True)
        csv_buffer = io.StringIO()
        combined_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{predictor.ticker}_predictions.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download_plot')
def download_plot():
    global predictor
    if predictor is None:
        return jsonify({'error': 'No model trained yet'}), 400
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        predictions = predictor.predict_future(30)
        plt.figure(figsize=(12, 6))
        plt.plot(predictor.data['Date'], predictor.data['Close'], label='Historical', color='blue')
        plt.plot(predictions['Date'], predictions['Close'], label='Predicted', color='red', linestyle='--')
        plt.title(f'{predictor.ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        plt.close()
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'{predictor.ticker}_prediction_plot.png'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def create_interactive_plot(data, predictions, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=predictions['Date'],
        y=predictions['Close'],
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    fig.update_layout(
        title=f'{ticker} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white'
    )
    return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

def calculate_metrics(data, predictions):
    if len(data) > 30:
        validation_data = data.tail(30)
        pred_values = predictions.head(30)['Close'].values
        actual_values = validation_data['Close'].values
        rmse = np.sqrt(np.mean((actual_values - pred_values) ** 2))
        mae = np.mean(np.abs(actual_values - pred_values))
        return {
            'rmse': round(rmse, 2),
            'mae': round(mae, 2)
        }
    return {
        'rmse': 'N/A',
        'mae': 'N/A'
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 