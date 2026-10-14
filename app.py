import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import tensorflow as tf

# Cố định seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

st.title('Ứng dụng Dự đoán Giá Cổ phiếu')

# Sidebar
st.sidebar.header('Cài đặt')
ticker = st.sidebar.text_input('Nhập mã chứng khoán (ví dụ: VNM.VN):', 'VNM.VN')
days_to_predict = st.sidebar.slider('Số ngày dự báo:', min_value=1, max_value=10, value=10)

# Hàm tải dữ liệu
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2018-01-01", end="2026-10-14")
    return df

data_load_state = st.text('Đang tải dữ liệu...')
df = load_data(ticker)
data_load_state.text('Đã tải dữ liệu xong!')

st.subheader('Dữ liệu Cổ phiếu thô')
st.write(df.tail())

# Xử lý dữ liệu
data = df[['Close']].values
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled)
test_combined = np.vstack((train_scaled[-60:], test_scaled))
X_test, y_test = create_sequences(test_combined)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình (giả lập để tiết kiệm thời gian, trong thực tế nên load model đã train)
st.text('Đang huấn luyện mô hình (có thể mất một chút thời gian)...')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
st.text('Huấn luyện xong!')

# Dự báo dữ liệu test
test_pred = model.predict(X_test)
test_pred = scaler.inverse_transform(test_pred)
y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))

# Đánh giá
rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
mae = mean_absolute_error(y_test_real, test_pred)
r2 = r2_score(y_test_real, test_pred)

st.subheader('Chỉ số đánh giá Mô hình')
st.write(f"RMSE: {rmse:.4f}")
st.write(f"MAE: {mae:.4f}")
st.write(f"R2: {r2:.4f}")

# Vẽ biểu đồ giá thật và dự báo
st.subheader('Biểu đồ Giá thật và Dự báo (Test Data)')
fig = plt.figure(figsize=(12,6))
plt.plot(y_test_real, label="Giá thật")
plt.plot(test_pred, label="Dự báo")
plt.legend()
st.pyplot(fig)

# Dự báo tương lai
last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
last_60_days = last_60_days.reshape(1, 60, 1)
future_predictions = []

for i in range(days_to_predict):
    next_pred = model.predict(last_60_days)
    future_predictions.append(next_pred[0,0])
    last_60_days = np.append(last_60_days[:,1:,:], next_pred.reshape(1,1,1), axis=1)

future_predictions = np.array(future_predictions).reshape(-1,1)
future_predictions = scaler.inverse_transform(future_predictions)

# Tính tỷ lệ lỗ/lãi
last_price = data[-1][0]
future_prices = future_predictions.flatten()
profit_loss = ((future_prices - last_price) / last_price) * 100

# Tạo DataFrame hiển thị dự báo
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
forecast_df = pd.DataFrame({
    'Ngày': future_dates,
    'Giá Dự Báo': future_prices,
    'Lỗ/Lãi (%)': profit_loss
})

st.subheader(f'Dự báo cho {days_to_predict} ngày tới')
st.dataframe(forecast_df)

# Biểu đồ dự báo tương lai
st.subheader('Biểu đồ Dự báo Tương lai')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test_real[-100:], label="Giá thật (100 ngày gần nhất)")
plt.plot(range(len(y_test_real[-100:]), len(y_test_real[-100:]) + days_to_predict), future_prices, marker='o', label="Dự báo tương lai")
plt.legend()
st.pyplot(fig2)
