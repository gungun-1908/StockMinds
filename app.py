import dash
import yfinance as yf
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dash import dcc, html
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
import requests

app = Flask(__name__)

def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1y")
    data['Date'] = data.index
    data['High-Low'] = data['High'] - data['Low']
    data['High-Open'] = data['High'] - data['Open']
    data['Low-Open'] = data['Low'] - data['Open']
    data['Close-Open'] = data['Close'] - data['Open']
    data = data.dropna()
    return data

def prepare_features_and_target(data, days_to_predict):
    X = data[['Open', 'High', 'Low', 'Volume', 'High-Low', 'High-Open', 'Low-Open', 'Close-Open']]
    y = data['Close'].shift(-days_to_predict)
    y = y.dropna()

    X = X[:-days_to_predict]
    return X, y

def train_and_predict(symbol, days_to_predict):
    data = fetch_stock_data(symbol)

    X, y = prepare_features_and_target(data, days_to_predict)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make the prediction for the next 'days_to_predict'
    predicted_price = model.predict(X_test[-1:])[0]

    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error (MAE) for {symbol} ({days_to_predict} days): {mae}")
    
    return predicted_price, mae

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    days = int(request.form['days'])

    # Train model and make prediction
    predicted_price, mae = train_and_predict(symbol, days)

    return render_template('index.html', symbol=symbol, predicted_price=predicted_price, mae=mae, days=days)


app1 = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

stocks = ['AAPL', 'GOOGL', 'AMZN', 'HDFCBANK.NS', 'TATAMOTORS.NS', 'ICICIBANK.NS', 'MSFT', 'JPM', 'META']

def get_stock_data(stocks):
    data = {}
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            hist = ticker.history(period="1d")  
            if not hist.empty:
                data[stock] = hist.tail(1)
            else:
                print(f"No data found for {stock}. It may be delisted.")
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")
    return data

def process_data(stock_data):
    processed_data = []
    for stock, data in stock_data.items():
        if not data.empty:
            row = {
                'Name': stock,
                'Value': round(data['Close'][0], 2),
                'Change': round(data['Close'][0] - data['Open'][0], 2),
                '% Change': round((data['Close'][0] - data['Open'][0]) / data['Open'][0] * 100, 2),
                'Open': round(data['Open'][0], 2),
                'High': round(data['High'][0], 2),
                'Low': round(data['Low'][0], 2),
            }
            processed_data.append(row)
    return pd.DataFrame(processed_data)

# Dash layout (this will be directly displayed in the Flask page)
app1.layout = html.Div(children=[
    html.H1("Stock Dashboard", style={'textAlign': 'center'}),

    dash_table.DataTable( 
        id='stock-table',
        columns=[
            {"name": "Name", "id": "Name"},
            {"name": "Value", "id": "Value"},
            {"name": "Change", "id": "Change"},
            {"name": "% Change", "id": "% Change"},
            {"name": "Open", "id": "Open"},
            {"name": "High", "id": "High"},
            {"name": "Low", "id": "Low"},
        ],
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'filter_query': '{Change} < 0', 'column_id': 'Change'}, 'backgroundColor': 'tomato', 'color': 'white'},
            {'if': {'filter_query': '{Change} > 0', 'column_id': 'Change'}, 'backgroundColor': 'green', 'color': 'white'},
        ],
    ),

    dcc.Interval(id='interval-component', interval=5*60*1000, n_intervals=0),  
])

# Callback to update table data
@app1.callback(
    Output('stock-table', 'data'), # type: ignore
    [Input('interval-component', 'n_intervals')] # type: ignore
)
def update_table(n):
    stock_data = get_stock_data(stocks)
    if not stock_data:
        raise PreventUpdate # type: ignore
    df = process_data(stock_data)
    return df.to_dict('records')


@app.route('/dashboard')
def dashboard():
    return app1.index()



NEWS_API_KEY = '7ed7a776d5d14c1fa5b7546178e0c50c'


def get_stock_news(stock_symbol):
    url = f'https://newsapi.org/v2/everything?q={stock_symbol}&language=en&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'ok':
        return data['articles'][:5]
    else:
        return []

@app.route("/news", methods=["GET","POST"])
def index1():
    news_articles = []
    if request.method == "POST":
        stock_symbol = request.form["stock_symbol"]
        news_articles = get_stock_news(stock_symbol)
    return render_template("index.html", news_articles=news_articles)


def get_stock_summary(symbol):
    stock = yf.Ticker(symbol)
    stock_info = stock.info

    try:
        stock_summary = {
            "symbol": stock_info.get("symbol", "N/A"),
            "name": stock_info.get("longName", "N/A"),
            "current_price": stock_info.get("currentPrice", "N/A"),
            "previous_close": stock_info.get("regularMarketPreviousClose", "N/A"),
            "market_cap": stock_info.get("marketCap", "N/A"),
            "pe_ratio": stock_info.get("trailingPE", "N/A"),
            "day_high": stock_info.get("dayHigh", "N/A"),
            "day_low": stock_info.get("dayLow", "N/A"),
            "fifty_two_week_high": stock_info.get("fiftyTwoWeekHigh", "N/A"),
            "fifty_two_week_low": stock_info.get("fiftyTwoWeekLow", "N/A"),
            "volume": stock_info.get("volume", "N/A"),
        }
    except Exception as e:
        stock_summary = {
            "error": f"Error retrieving data for {symbol}: {str(e)}"
        }
    return stock_summary

@app.route("/summarize", methods=["POST"])
def index2():
    stock_summary = None
    if request.method == "POST":
        symbol = request.form["stock_symbol"]
        stock_summary = get_stock_summary(symbol)
    return render_template("index.html", stock_summary=stock_summary)

if __name__ == "__main__":
    app.run(debug=True)
