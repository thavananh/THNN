# Phân Loại Email CNN-BiLSTM-Attention

Ứng dụng **Phân Tích email Bằng kiến trúc XLMRoBerta-CNN-BiLSTM** là một ứng dụng web được xây dựng bằng Streamlit, sử dụng mô hình huấn luyện trước XLMRoBerta cùng với các kỹ thuật học sâu kết hợp giữa Convolutional Neural Network (CNN), Bidirectional LSTM (BiLSTM) để phân loại email thành hai loại: **Spam** và **Ham**

## Tính Năng

- **Phân loại email**: Nhập văn bản tiếng Việt hoặc tiếng anh anh và phân loại email tương ứng.
- **Giao diện thân thiện**: Sử dụng Streamlit để tạo giao diện đơn giản và dễ sử dụng.
- **Ví dụ mẫu**: Hiển thị ví dụ phân loại khi người dùng muốn.

## Yêu Cầu Hệ Thống

- **Python**: Phiên bản 3.7 trở lên
- **Các thư viện Python**: Xem mục [Cài Đặt](#cài-đặt) để biết chi tiết.

## Cài Đặt

1. **Clone kho lưu trữ**:

   ```bash
   git clone https://github.com/thavananh/THNN.git
   cd THNN
   ```

2. **Tạo môi trường ảo (khuyến nghị)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Windows: venv\Scripts\activate
   ```

3. **Cài đặt các thư viện phụ thuộc qua `requirements.txt`**:

   Đảm bảo bạn đã có tệp `requirements.txt` trong thư mục dự án. Tệp này liệt kê tất cả các thư viện cần thiết để chạy ứng dụng. Nội dung mẫu của `requirements.txt`:

   ```plaintext
   streamlit
   tensorflow
   pyvi
   numpy
   torch
   seaborn
   transformers
   streamlit-plugins
   ```

   Sau đó, chạy lệnh sau để cài đặt tất cả các thư viện:

   ```bash
   pip install -r requirements.txt
   ```

4. **Tải mô hình và tokenizer**:

   Đảm bảo bạn có các thư mục sau trong thư mục dự án:

   - `cnn_lstm_attention_component`: Thư mục mô hình CNN_LSTM, XLM-Roberta-CNN-LSTM.
   - `ML_archive`: Thư mục mô hình machine learning.

   Nếu bạn chưa có các tệp này, hãy liên hệ với tác giả hoặc kiểm tra tài liệu đi kèm để biết cách tạo và lưu trữ chúng.

## Sử Dụng

### Chạy Ứng Dụng

Để chạy ứng dụng Streamlit, sử dụng lệnh sau:

```bash
streamlit run app.py
```
