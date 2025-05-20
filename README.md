# Spam mail detection using machine learning
Với bối cảnh tăng trưởng người dùng sử dụng Internet, các mail, tin nhắn rác cũng ngày một gia tăng với các thủ đoạn ngày một tinh vi. Các mail, tin nhắn này được sử dụng cho các mục đích xấu như lừa đảo hay chiếm đạt thông tin. Vì vậy mà việc xác định và nhận diện các mail spam, mail lừa đảo này là cần thiết, chính vì thế mà nhóm quyết định áp dụng máy học vào giải quyết vấn đề này.

Mục tiêu của đề tài/dự án này là giải quyết được vấn đề nhận diện, phân loại các loại mail xem đâu là mail lừa đảo và đâu là mail bình thường. Việc phân loại được các mail, tin nhắn rác sẽ giúp tiết kiệm được thời gian, tiền, và không gian kho lưu trữ. Đề tài cũng hi vọng sẽ phần nào giúp biến Internet trở thành một nơi an toàn hơn cho tất cả mọi người và nâng cao tính cảnh giác đối với các loại thư văn bản được gửi tới người đọc.

## 📋 Tổng quan
Dự án này tập trung vào việc phát hiện gian lận thông qua phân loại email thành "Spam" hoặc "Ham" (không phải spam) bằng cách sử dụng mô hình học máy. Việc phát hiện email spam là một ứng dụng quan trọng trong lĩnh vực bảo mật thông tin và phòng chống lừa đảo.

---

## 📂 Các tệp trong dự án

### 1. **`emails.csv`**
- **Mô tả**: Tập dữ liệu được lấy từ Kaggle, chứa thông tin về các email dùng để huấn luyện và kiểm tra mô hình.
- **Đường dẫn**: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
- **Mục đích**: Sử dụng để huấn luyện mô hình học máy và phân tích dữ liệu email.

---

### 2. **`train_model.py`**
- **Mô tả**: Chương trình Python dùng để tiền xử lý dữ liệu và huấn luyện mô hình AI.
- **Chức năng chính**:
  - Tải và tiền xử lý dữ liệu từ `emails.csv`.
  - Chuẩn hóa dữ liệu bằng `MinMaxScaler`.
  - Chia dữ liệu thành tập huấn luyện và kiểm tra.
  - Xây dựng và huấn luyện mô hình mạng nơ-ron sử dụng TensorFlow/Keras.
  - Lưu mô hình đã huấn luyện (`spam_classifier_model.keras`) và scaler (`scaler.pkl`) để sử dụng sau này.

---

### 3. **`get_keywords.py`**
- **Mô tả**: Chương trình Python dùng để trích xuất từ khóa từ tập dữ liệu.
- **Chức năng chính**:
  - Đọc dữ liệu từ `emails.csv`.
  - Trích xuất 3.000 từ khóa tương ứng với các cột trong tập dữ liệu.
  - Lưu danh sách từ khóa vào tệp `keywords.json`.

---

### 4. **`test_model.py`**
- **Mô tả**: Chương trình Python dùng để kiểm tra mô hình đã huấn luyện trên các email mẫu.
- **Chức năng chính**:
  - Tải mô hình đã huấn luyện (`spam_classifier_model.keras`) và scaler (`scaler.pkl`).
  - Đọc danh sách từ khóa từ `keywords.json`.
  - Xử lý các email mẫu từ `email_test.json`.
  - Dự đoán email là "Spam" hay "Ham" dựa trên mô hình đã huấn luyện.

---

### 5. **`keywords.json`**
- **Mô tả**: Tệp JSON lưu trữ danh sách các từ khóa đã trích xuất.
- **Mục đích**: Được sử dụng bởi tệp `test_model.py` để chuyển đổi email thành vector đặc trưng phục vụ dự đoán.

---

### 6. **`email_test.json`**
- **Mô tả**: Tệp JSON chứa các email mẫu để kiểm thử mô hình.
- **Mục đích**: Cung cấp dữ liệu đầu vào cho tệp `test_model.py` để đánh giá hiệu suất của mô hình.

---

### 7. **`requirements.txt`**
- **Mô tả**: Tệp chứa danh sách các thư viện Python cần thiết cho dự án.
- **Mục đích**: Giúp dễ dàng cài đặt các dependency cần thiết bằng lệnh `pip install -r requirements.txt`.

---

### 8. **`scaler.pkl`**
- **Mô tả**: Tệp lưu trữ đối tượng `MinMaxScaler` đã được huấn luyện.
- **Mục đích**: Sử dụng để chuẩn hóa dữ liệu đầu vào trong cả giai đoạn huấn luyện và kiểm tra.

---

### 9. **`spam_classifier_model.keras`**
- **Mô tả**: Mô hình học máy đã được huấn luyện, lưu dưới định dạng TensorFlow/Keras.
- **Mục đích**: Được sử dụng bởi tệp `test_model.py` để phân loại email.

---

### 10. **`eda-ppnhan.py`**
- **Mô tả**: Script phân tích khám phá dữ liệu (Exploratory Data Analysis).
- **Chức năng chính**:
  - Phân tích phân phối nhãn Ham/Spam trong tập dữ liệu
  - Trực quan hóa Top 20 từ xuất hiện nhiều nhất trong email Spam và Ham
  - Lưu các biểu đồ vào thư mục `eda_images/`

---

### 11. **`eda_images/`**
- **Mô tả**: Thư mục chứa các biểu đồ trực quan hóa từ phân tích dữ liệu.
- **Các file chính**:
  - `label_distribution.png` - Biểu đồ phân phối nhãn Spam/Ham
  - `top_words_spam.png` - Top 20 từ xuất hiện nhiều nhất trong email Spam
  - `top_words_ham.png` - Top 20 từ xuất hiện nhiều nhất trong email Ham

---

## 🚀 Hướng dẫn sử dụng

1. **Huấn luyện mô hình**:
   - Chạy tệp `train_model.py` để tiền xử lý dữ liệu, huấn luyện mô hình và lưu mô hình cùng scaler.

2. **Trích xuất từ khóa**:
   - Chạy tệp `get_keywords.py` để trích xuất từ khóa từ tập dữ liệu và lưu vào `keywords.json`.

3. **Kiểm tra mô hình**:
   - Chạy tệp `test_model.py` để tải mô hình đã huấn luyện và kiểm tra trên các email mẫu từ `email_test.json`.

---

## 📄 Ghi chú
- Đảm bảo cài đặt đầy đủ các thư viện cần thiết trước khi chạy các file Python.
- Nếu thay đổi cấu trúc thư mục, cần cập nhật lại đường dẫn trong các tệp file Python.

---

## 🛠️ Các thư viện cần thiết
- Python 3.x
- TensorFlow
- scikit-learn
- pandas
- numpy
- joblib

---

## 🛠️ Cách cài thư viện
- Mở Terminal, nhập:
  ```
  pip install -r requirements.txt
  ```
  
© Nghiên cứu khoa học - Trường Đại học Sài Gòn (SGU)
