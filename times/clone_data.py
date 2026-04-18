import yfinance as yf
import pandas as pd

OUTPUT_FILE = "external_macro_data.csv"

def get_external_data():
    # Tải từ đầu năm 2019 để làm 'vùng đệm' tính Lag (độ trễ)
    start_date = "2019-01-01"
    end_date = "2019-07-01"

    # BZ=F: Brent Crude Oil (Dầu Brent)
    # INR=X: USD to Indian Rupee (Tỷ giá)
    tickers = ["BZ=F", "INR=X"]

    print("--- Đang tải dữ liệu từ Yahoo Finance ---")

    # Tải dữ liệu (chỉ lấy giá đóng cửa - Close)
    df = yf.download(tickers, start=start_date, end=end_date)["Close"]

    # Đổi tên cột cho rõ nghĩa
    df = df.rename(columns={
        "BZ=F": "Brent_Oil_Price",
        "INR=X": "USD_INR_Exchange",
    })

    # Dữ liệu tài chính chỉ có Thứ 2 – Thứ 6.
    # .asfreq('D') tạo dòng cho Thứ 7, CN còn thiếu
    # .ffill() lấy giá Thứ 6 điền vào Thứ 7 và CN
    df_full = df.asfreq("D").ffill()

    # Reset index để cột Date trở thành cột dữ liệu bình thường
    df_full = df_full.reset_index()

    print(f"Tải thành công! Số lượng bản ghi: {len(df_full)}")
    return df_full


def save_data(df: pd.DataFrame, path: str = OUTPUT_FILE) -> None:
    df.to_csv(path, index=False)
    print(f"Đã lưu dữ liệu vào '{path}'")


if __name__ == "__main__":
    external_data = get_external_data()
    print(external_data.head(10))
    save_data(external_data)
