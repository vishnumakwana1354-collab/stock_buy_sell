import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Stock Buy/Sell Prediction",
    page_icon="📈",
    layout="wide"
)

# ==========================
# Title
# ==========================
st.title("📈 Stock Buy/Sell Prediction Dashboard")
st.markdown("Predict **BUY** or **SELL** signals using a trained KNN Machine Learning model.")

# ==========================
# Sidebar
# ==========================
st.sidebar.header("Settings")

stock = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
)

period = st.sidebar.selectbox(
    "Select Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

load_data = st.sidebar.button("📥 Load Stock Data")

# ==========================
# Load KNN Model
# ==========================
knn = None
scaler = None

try:
    knn = joblib.load("models/knn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    st.sidebar.success("✅ Model Loaded Successfully")
except Exception as e:
    st.sidebar.warning("⚠ Model not loaded")
    st.sidebar.caption(str(e))

# ==========================
# Download Stock Data
# ==========================
if load_data:

    with st.spinner("Downloading stock data..."):

        df = yf.download(stock, period=period)

    # Flatten MultiIndex columns if required
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        st.error("No data found.")
        st.stop()

    # ==========================
    # Latest Price
    # ==========================
    latest_close = float(df["Close"].iloc[-1])
    latest_open = float(df["Open"].iloc[-1])
    latest_high = float(df["High"].iloc[-1])
    latest_low = float(df["Low"].iloc[-1])

    # ==========================
    # Metrics
    # ==========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Close", f"${latest_close:.2f}")
    col2.metric("Open", f"${latest_open:.2f}")
    col3.metric("High", f"${latest_high:.2f}")
    col4.metric("Low", f"${latest_low:.2f}")

    # ==========================
    # Price Chart
    # ==========================
    st.subheader("📈 Closing Price")

    st.line_chart(df["Close"])

    # ==========================
    # Volume Chart
    # ==========================
    st.subheader("📊 Trading Volume")

    st.bar_chart(df["Volume"])

    # ==========================
    # Latest Data
    # ==========================
    st.subheader("Latest Stock Data")

    st.dataframe(df.tail())

    # =====================================
    # Feature Engineering
    # =====================================

    # Create features
    df["Price_Change"] = df["Close"] - df["Open"]
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["Daily_Return"] = df["Close"].pct_change()

    # Remove missing values
    df.dropna(inplace=True)

    # =====================================
    # Prediction
    # =====================================

    st.subheader("🤖 Buy / Sell Prediction")

    if knn is not None and scaler is not None:

        features = df[[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Price_Change",
            "MA5",
            "MA10",
            "Daily_Return"
        ]]

        # Take latest row
        latest_data = features.tail(1)

        # Scale data
        latest_scaled = scaler.transform(latest_data)

        # Prediction
        prediction = knn.predict(latest_scaled)[0]

        if prediction == 1:
            st.success("🟢 BUY SIGNAL")
        else:
            st.error("🔴 SELL SIGNAL")

    else:
        st.warning("Model not loaded.")

    st.subheader("📊 Model Information")

    st.info("""
Machine Learning Model : K-Nearest Neighbors (KNN)

Features Used:
- Open
- High
- Low
- Close
- Volume
- Price Change
- MA5
- MA10
- Daily Return
""")