import pandas as pd
import json

# --- Thông tin file ---
CSV_FILE = 'emails.csv'    # Đổi tên file nếu cần
KEYWORDS_FILE = 'keywords.json'

print("Load file emails.csv...")

# Load dữ liệu
df = pd.read_csv(CSV_FILE)
print(f"File có {df.shape[0]} dòng và {df.shape[1]} cột.")

# --- Xử lý từ khóa ---
print("🔍 Trích danh sách từ khóa...")

# Danh sách cột
columns = list(df.columns)

# Bỏ những cột không phải từ khóa
columns_to_drop = ['Email No.', 'Prediction', 'label', 'Label']

# Giữ lại các cột là từ khóa
keywords = [col for col in columns if col not in columns_to_drop]

print(f"📈 Số từ khóa trích được: {len(keywords)}")

# Lưu danh sách từ khóa
with open(KEYWORDS_FILE, "w", encoding="utf-8") as f:
    json.dump(keywords, f, indent=4, ensure_ascii=False)

print(f"📄 File {KEYWORDS_FILE} đã được lưu hoàn tất!")
