import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

# Constants
start = '2012-01-01'
Today = date.today().strftime('%Y-%m-%d')

# Streamlit App Title
st.title('Stock Predictor Extension')

# Stock Selection
stocks = ('GOOG', 'AAPL', 'INTC', 'NVDA', 'MSFT', 'TSLA', 'RELIANCE.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'ADANIENT.NS')
selected_stock = st.selectbox('Select a Stock for Prediction', stocks)

# User Stock Input
user_input = st.text_input('Or Enter a Stock for Prediction (e.g., AAPL)')

# Determine the ticker to use
ticker = user_input if user_input else selected_stock

# Years of Prediction
n_years = st.slider('Years of Prediction:', 1, 5)
period = n_years * 365

# Data Loading with Caching
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, Today)
    data.reset_index(inplace=True)
    return data

# Load Data
data_load_state = st.text('Loading data...')
data = load_data(ticker)
data_load_state.text('Loading data... done!')

# Display Raw Data
st.subheader('Raw Data')
st.write(data.tail())

# Plot Raw Data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text='Time Series', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare Data for Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

# Forecasting
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display Forecast Data
st.subheader('Forecast Data')
st.write(forecast.tail())

# Plot Forecast
st.write('Forecast Plot')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Forecast Components
st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)
