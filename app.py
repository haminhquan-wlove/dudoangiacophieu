
# ============================================================
# APP DỰ BÁO GIÁ CỔ PHIẾU BẰNG STREAMLIT VÀ LSTM
# File: app.py
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import io
import time

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -------------------- CẤU HÌNH --------------------

st.set_page_config(
    page_title="Dự báo giá cổ phiếu AI",
    page_icon="📈",
    layout="wide"
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -------------------- GIAO DIỆN --------------------

st.title("📈 ỨNG DỤNG DỰ BÁO GIÁ CỔ PHIẾU AI")
st.markdown(
    """
    Theo dõi giá cổ phiếu, dự báo xu hướng trong tối đa 10 ngày
    giao dịch và ước tính mức tăng/giảm giá dự kiến.
    """
)

st.warning(
    "Dự báo chỉ mang tính tham khảo, không phải lời khuyên đầu tư. "
    "Giá thực tế có thể khác đáng kể so với kết quả mô hình."
)

# -------------------- HÀM HỖ TRỢ --------------------

@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(ticker, period):
    """Tải dữ liệu giá lịch sử từ Yahoo Finance."""
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Xử lý dữ liệu có nhiều cấp cột do yfinance trả về
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        return pd.DataFrame()

    df = df[["Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    df = df[df["Close"] > 0]
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    return df


def create_sequences(data, window_size=60):
    """Tạo chuỗi dữ liệu đầu vào cho LSTM."""
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X.reshape(-1, window_size, 1), y


def build_model(window_size=60):
    """Khởi tạo mô hình LSTM."""
    model = Sequential([
        LSTM(
            50,
            return_sequences=True,
            input_shape=(window_size, 1)
        ),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error"
    )

    return model


def forecast_stock(df, horizon, progress_callback=None):
    """
    Huấn luyện mô hình trên dữ liệu lịch sử rồi dự báo
    lần lượt các phiên giao dịch tiếp theo.
    """
    window_size = 60
    values = df["Close"].values.reshape(-1, 1)

    if len(values) < 150:
        raise ValueError(
            "Không đủ dữ liệu lịch sử. Hãy chọn khoảng thời gian dài hơn."
        )

    # Chia dữ liệu theo thứ tự thời gian, không xáo trộn
    train_size = int(len(values) * 0.85)
    train_values = values[:train_size]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)

    X_train, y_train = create_sequences(
        train_scaled,
        window_size
    )

    if len(X_train) < 20:
        raise ValueError("Số lượng dữ liệu huấn luyện quá ít.")

    # Dùng phần cuối tập huấn luyện để theo dõi quá trình học
    val_size = max(1, int(len(X_train) * 0.1))

    X_fit = X_train[:-val_size]
    y_fit = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    if len(X_fit) == 0:
        raise ValueError("Không đủ dữ liệu để huấn luyện mô hình.")

    model = build_model(window_size)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # Đánh giá trên tập kiểm tra ở cuối chuỗi thời gian
    test_scaled = scaler.transform(values[train_size:])
    combined = np.vstack((train_scaled[-window_size:], test_scaled))

    X_test, y_test = create_sequences(
        combined,
        window_size
    )

    if len(X_test) > 0:
        pred_scaled = model.predict(X_test, verbose=0)
        test_pred = scaler.inverse_transform(pred_scaled).ravel()
        actual = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).ravel()

        mae = mean_absolute_error(actual, test_pred)
        rmse = np.sqrt(mean_squared_error(actual, test_pred))
    else:
        mae = np.nan
        rmse = np.nan

    # Huấn luyện lại trên toàn bộ dữ liệu để dự báo tương lai
    all_scaled = scaler.fit_transform(values)

    X_all, y_all = create_sequences(
        all_scaled,
        window_size
    )

    final_model = build_model(window_size)

    final_model.fit(
        X_all,
        y_all,
        epochs=15,
        batch_size=32,
        shuffle=False,
        verbose=0
    )

    # Dự báo cuốn chiếu: mỗi dự báo được đưa vào chuỗi kế tiếp
    last_window = all_scaled[-window_size:].reshape(
        1, window_size, 1
    )

    predictions_scaled = []

    for step in range(horizon):
        next_scaled = final_model.predict(
            last_window,
            verbose=0
        )[0, 0]

        predictions_scaled.append(next_scaled)

        next_input = np.array(
            [[[next_scaled]]],
            dtype=np.float32
        )

        last_window = np.concatenate(
            [last_window[:, 1:, :], next_input],
            axis=1
        )

        if progress_callback:
            progress_callback((step + 1) / horizon)

    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).ravel()

    # Ngày giao dịch dự kiến: bỏ thứ Bảy và Chủ nhật
    last_date = pd.Timestamp(df.index[-1]).normalize()
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon
    )

    forecast_df = pd.DataFrame({
        "Ngày dự báo": future_dates,
        "Giá dự báo": predictions
    })

    return forecast_df, mae, rmse, test_pred if len(X_test) > 0 else [], (
        actual if len(X_test) > 0 else []
    )


def format_price(value, ticker):
    """Định dạng giá theo thị trường."""
    if ticker.upper().endswith(".VN"):
        return f"{value:,.0f} VND"
    return f"{value:,.2f} USD"


# -------------------- THANH ĐIỀU KHIỂN --------------------

st.sidebar.header("⚙️ Thiết lập dự báo")

market = st.sidebar.selectbox(
    "Thị trường",
    ["Việt Nam", "Mỹ / Quốc tế"]
)

if market == "Việt Nam":
    popular_stocks = [
        "VNM", "FPT", "HPG", "VCB", "VIC",
        "MWG", "SSI", "CTG", "TCB", "GAS"
    ]
    selected = st.sidebar.selectbox(
        "Chọn mã cổ phiếu phổ biến",
        ["Tự nhập mã"] + popular_stocks
    )
    if selected == "Tự nhập mã":
        ticker_input = st.sidebar.text_input(
            "Nhập mã chứng khoán",
            value="VNM"
        )
    else:
        ticker_input = selected

    ticker = ticker_input.strip().upper()
    if ticker and "." not in ticker and "^" not in ticker:
        ticker += ".VN"

else:
    popular_stocks = [
        "AAPL", "MSFT", "NVDA", "TSLA",
        "AMZN", "GOOGL", "META", "SPY"
    ]
    selected = st.sidebar.selectbox(
        "Chọn mã cổ phiếu phổ biến",
        ["Tự nhập mã"] + popular_stocks
    )
    if selected == "Tự nhập mã":
        ticker_input = st.sidebar.text_input(
            "Nhập mã chứng khoán",
            value="AAPL"
        )
    else:
        ticker_input = selected

    ticker = ticker_input.strip().upper()

period_label = st.sidebar.selectbox(
    "Khoảng dữ liệu lịch sử",
    [
        "2 năm",
        "5 năm",
        "10 năm"
    ],
    index=1
)

period_map = {
    "2 năm": "2y",
    "5 năm": "5y",
    "10 năm": "10y"
}
period = period_map[period_label]

horizon = st.sidebar.slider(
    "Dự báo bao nhiêu ngày giao dịch?",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)

st.sidebar.caption(
    "Mô hình sử dụng giá đóng cửa đã điều chỉnh và dữ liệu lịch sử."
)

run_button = st.sidebar.button(
    "🚀 BẮT ĐẦU DỰ BÁO",
    type="primary",
    use_container_width=True
)

# -------------------- NỘI DUNG CHÍNH --------------------

st.subheader("📌 Thông tin dự báo")
st.write(f"**Mã được chọn:** `{ticker}`")
st.write(f"**Thời gian dự báo:** {horizon} ngày giao dịch")

if run_button:
    if not ticker:
        st.error("Vui lòng nhập mã chứng khoán.")
        st.stop()

    try:
        with st.spinner("Đang tải dữ liệu giá cổ phiếu..."):
            df = get_stock_data(ticker, period)

        if df.empty:
            st.error(
                "Không tìm thấy dữ liệu cho mã này. "
                "Hãy kiểm tra mã chứng khoán hoặc thử lại sau."
            )
            st.stop()

        if len(df) < 150:
            st.error(
                "Dữ liệu lịch sử chưa đủ để huấn luyện mô hình. "
                "Hãy chọn khoảng thời gian dài hơn hoặc mã khác."
            )
            st.stop()

        current_price = float(df["Close"].iloc[-1])
        last_date = df.index[-1]

        st.success(
            f"Đã tải {len(df):,} phiên dữ liệu. "
            f"Phiên gần nhất: {pd.Timestamp(last_date).strftime('%d/%m/%Y')}."
        )

        progress_bar = st.progress(0, text="Đang huấn luyện mô hình LSTM...")

        def update_progress(value):
            progress_bar.progress(
                min(int(value * 100), 100),
                text=f"Đang tạo dự báo: {int(value * 100)}%"
            )

        with st.spinner("Đang huấn luyện và dự báo..."):
            forecast_df, mae, rmse, test_pred, actual = forecast_stock(
                df,
                horizon,
                progress_callback=update_progress
            )

        progress_bar.progress(100, text="Hoàn tất dự báo!")

        final_price = float(forecast_df["Giá dự báo"].iloc[-1])
        change = final_price - current_price
        change_pct = (
            (final_price / current_price - 1) * 100
            if current_price != 0 else np.nan
        )

        # -------------------- CÁC CHỈ SỐ --------------------

        st.divider()
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Giá đóng cửa gần nhất",
            format_price(current_price, ticker)
        )

        col2.metric(
            f"Dự báo ngày thứ {horizon}",
            format_price(final_price, ticker)
        )

        col3.metric(
            "Chênh lệch giá dự kiến",
            format_price(change, ticker),
            delta=f"{change_pct:+.2f}%"
        )

        col4.metric(
            "MAE kiểm tra",
            format_price(mae, ticker) if np.isfinite(mae) else "N/A"
        )

        if change_pct > 0:
            st.success(
                f"Theo mô hình, giá tại ngày dự báo cuối cao hơn "
                f"giá đóng cửa gần nhất khoảng {change_pct:.2f}%."
            )
        elif change_pct < 0:
            st.warning(
                f"Theo mô hình, giá tại ngày dự báo cuối thấp hơn "
                f"giá đóng cửa gần nhất khoảng {abs(change_pct):.2f}%."
            )
        else:
            st.info("Mô hình dự báo giá cuối kỳ gần bằng giá hiện tại.")

        st.caption(
            "MAE là sai số tuyệt đối trung bình trên tập kiểm tra theo thời gian; "
            "sai số quá khứ không đảm bảo độ chính xác của dự báo tương lai."
        )

        # -------------------- BIỂU ĐỒ --------------------

        st.subheader("📊 Biểu đồ giá lịch sử và dự báo")

        history_to_show = df.tail(180)

        fig, ax = plt.subplots(figsize=(13, 5))

        ax.plot(
            history_to_show.index,
            history_to_show["Close"],
            label="Giá lịch sử",
            linewidth=2
        )

        # Nối giá cuối cùng với chuỗi dự báo
        forecast_dates = pd.DatetimeIndex(
            [pd.Timestamp(last_date)] +
            list(forecast_df["Ngày dự báo"])
        )
        forecast_prices = np.concatenate((
            [current_price],
            forecast_df["Giá dự báo"].values
        ))

        ax.plot(
            forecast_dates,
            forecast_prices,
            marker="o",
            linestyle="--",
            label="Giá dự báo LSTM",
            linewidth=2
        )

        ax.axvline(
            pd.Timestamp(last_date),
            linestyle=":",
            label="Bắt đầu dự báo"
        )

        ax.set_title(f"Dự báo giá cổ phiếu {ticker}")
        ax.set_xlabel("Ngày")
        ax.set_ylabel(
            "Giá (VND)" if ticker.endswith(".VN") else "Giá (USD)"
        )
        ax.legend()
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # -------------------- BẢNG DỰ BÁO --------------------

        st.subheader("📋 Bảng dự báo chi tiết")

        result_df = forecast_df.copy()
        result_df["Ngày dự báo"] = pd.to_datetime(
            result_df["Ngày dự báo"]
        ).dt.strftime("%d/%m/%Y")

        result_df["Giá dự báo"] = result_df["Giá dự báo"].round(2)

        result_df["Thay đổi so với giá hiện tại (%)"] = (
            (forecast_df["Giá dự báo"] / current_price - 1) * 100
        ).round(2)

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------- TẢI CSV --------------------

        csv_buffer = io.StringIO()
        result_df.to_csv(
            csv_buffer,
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            label="⬇️ Tải bảng dự báo CSV",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )

        # -------------------- LỊCH SỬ GIÁ --------------------

        with st.expander("Xem dữ liệu giá lịch sử"):
            st.dataframe(
                df.tail(100).sort_index(ascending=False),
                use_container_width=True
            )

        st.info(
            "Lưu ý: ngày dự báo loại trừ thứ Bảy và Chủ nhật, "
            "nhưng chưa loại trừ các ngày nghỉ lễ của từng thị trường. "
            "Dự báo nhiều bước có thể tích lũy sai số."
        )

    except Exception as e:
        st.error(
            "Không thể hoàn thành dự báo. "
            "Hãy kiểm tra kết nối Internet, mã chứng khoán và thư viện đã cài đặt."
        )
        st.exception(e)

else:
    st.info(
        "Chọn thị trường, nhập mã cổ phiếu, chọn khoảng dữ liệu "
        "và số ngày dự báo ở thanh bên trái. Sau đó bấm "
        "**BẮT ĐẦU DỰ BÁO**."
    )

    st.subheader("🧭 Cách sử dụng")
    st.markdown(
        """
        1. Chọn thị trường Việt Nam hoặc Mỹ / Quốc tế.
        2. Chọn mã có sẵn hoặc nhập mã khác.
        3. Chọn dữ liệu lịch sử 2, 5 hoặc 10 năm.
        4. Chọn số ngày dự báo từ 1 đến 10.
        5. Bấm **BẮT ĐẦU DỰ BÁO**.
        6. Xem giá dự báo, tỷ lệ tăng/giảm, biểu đồ và tải CSV.
        """
    )
