import pandas as pd

OUTPUT_FILE = "master_data_merged.csv"

# ─────────────────────────────────────────────
# 1. Đọc file Excel
# ─────────────────────────────────────────────
print("--- Đọc Data_Train.xlsx ---")
df = pd.read_excel("time series/Data_Train.xlsx")

# ─────────────────────────────────────────────
# 2. Chuyển đổi kiểu dữ liệu datetime
# ─────────────────────────────────────────────
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
df["Dep_Time"]        = pd.to_datetime(df["Dep_Time"], format="%H:%M", errors="coerce")

# ─────────────────────────────────────────────
# 3. Lọc chặng bay phổ biến nhất: Delhi → Cochin
# ─────────────────────────────────────────────
route = df[(df["Source"] == "Delhi") & (df["Destination"] == "Cochin")].copy()
print(f"Số chuyến bay Delhi → Cochin: {len(route)}")

# ─────────────────────────────────────────────
# 4. Gom nhóm theo ngày, lấy median giá vé
#    → Daily Flight Fare time series
# ─────────────────────────────────────────────
daily_fare = (
    route.groupby("Date_of_Journey", as_index=False)["Price"]
    .median()
    .rename(columns={"Date_of_Journey": "Date", "Price": "Median_Price"})
)
print(f"Số ngày có dữ liệu giá vé: {len(daily_fare)}")

# ─────────────────────────────────────────────
# 5. Đọc dữ liệu vĩ mô (giá dầu & tỷ giá)
# ─────────────────────────────────────────────
print("\n--- Đọc external_macro_data.csv ---")
macro = pd.read_csv("external_macro_data.csv")
macro["Date"] = pd.to_datetime(macro["Date"])

# Đảm bảo dữ liệu vĩ mô liên tục 7 ngày/tuần (ffill Thứ 7, CN)
macro = macro.set_index("Date").asfreq("D").ffill().reset_index()

# ─────────────────────────────────────────────
# 6. Ghép bảng giá vé với bảng vĩ mô theo cột Date
# ─────────────────────────────────────────────
master = pd.merge(daily_fare, macro, on="Date", how="left")
master = master.sort_values("Date").reset_index(drop=True)

print("\nKết quả (10 dòng đầu):")
print(master.head(10).to_string())
print(f"\nTổng số dòng: {len(master)} | Cột: {master.columns.tolist()}")

# ─────────────────────────────────────────────
# 7. Lưu kết quả ra file sạch
# ─────────────────────────────────────────────
master.to_csv(OUTPUT_FILE, index=False)
print(f"\nĐã lưu kết quả vào '{OUTPUT_FILE}'")
