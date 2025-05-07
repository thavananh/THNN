# Phân Loại Email XLMRoBerta-CNN-BiLSTM

Ứng dụng **Phân tích email bằng kiến trúc XLMRoBerta-CNN-BiLSTM** là một ứng dụng web được xây dựng bằng Streamlit, sử dụng mô hình huấn luyện trước XLMRoBerta cùng với các kỹ thuật học sâu kết hợp giữa Convolutional Neural Network (CNN) và Bidirectional LSTM (BiLSTM) để phân loại email thành hai loại: **Spam** và **Ham**.

## Tính Năng

* **Phân loại email**: Nhập văn bản tiếng Việt hoặc tiếng Anh và phân loại email tương ứng.
* **Giao diện thân thiện**: Sử dụng Streamlit để tạo giao diện đơn giản và dễ sử dụng.
* **Ví dụ mẫu**: Hiển thị ví dụ phân loại khi người dùng muốn.

## Yêu Cầu Hệ Thống

* **Python**: Phiên bản 3.7 trở lên.
* **Các thư viện Python**: Xem mục [Cài Đặt](#cài-đặt) để biết chi tiết.

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

   ```plaintext
   streamlit
   tensorflow
   pyvi
   numpy
   torch
   seaborn
   transformers
   ```

   ```bash
   pip install -r requirements.txt
   ```

4. **Tải mô hình và tokenizer**:

   Đảm bảo bạn có các thư mục sau trong thư mục dự án:

   * `cnn_lstm_attention_component`: Thư mục mô hình CNN-LSTM-Attention.
   * `xlm_roberta_cnn_lstm`: Thư mục mô hình XLMRoBerta\_CNN\_LSTM.

   Nếu bạn chưa có các tệp này, hãy tham khảo liên kết bên dưới để tải về.

## Demo Video và Link Các Mô Hình

* **Video Demo**: Xem video hướng dẫn và demo ứng dụng tại:

  [![Demo Video](/readme_resources/Thumnail_For_Video.png)](/readme_resources/demo_app.mkv)

* **Link Google Drive chứa mô hình**:

  * Tải xuống các mô hình ở đây: [https://drive.google.com/drive/folders/1Rguio6-Baq\_83cymRkDmHR9AcsbYQ\_Bo?usp=sharing](https://drive.google.com/drive/folders/1Rguio6-Baq_83cymRkDmHR9AcsbYQ_Bo?usp=sharing)

## Đánh giá mô hình

### Biểu đồ phân phối tập dữ liệu

![Biểu đồ phân phối tập dữ liệu](path/to/dataset_distribution.png)

### Ma trận nhầm lẫn (Confusion Matrix)

![Ma trận nhầm lẫn](path/to/confusion_matrix.png)

### Bảng đánh giá hiệu năng

#### Tập dữ liệu tiếng Việt (%)

| Model                        | Feature            | Precision | Recall | F1-Score |
| ---------------------------- | ------------------ | --------- | ------ | -------- |
| Logistic Regression          | TF-IDF             | 97.08     | 97.08  | 97.08    |
| Random Forest                | TF-IDF             | 96.84     | 96.82  | 96.82    |
| XGBoost                      | TF-IDF             | 97.48     | 97.48  | 97.48    |
| Naive Bayes                  | TF-IDF             | 93.83     | 93.63  | 93.62    |
| Extra Trees                  | TF-IDF             | 97.15     | 97.12  | 97.12    |
| LightGBM                     | TF-IDF             | 97.54     | 97.53  | 97.53    |
| Voting Classifier            | TF-IDF             | 97.77     | 97.77  | 97.77    |
| CNN-LSTM Attention           | Word2Vec Embedding | 98.04     | 98.04  | 98.04    |
| Bert-base-multilingual-cased | -                  | 98.97     | 98.97  | 98.97    |
| FPT-velectra                 | -                  | 98.82     | 98.82  | 98.82    |
| Distill-BERT-CNN-LSTM        | -                  | 98.65     | 98.65  | 98.65    |
| PhoBERT-CNN-LSTM             | -                  | 98.84     | 98.84  | 98.84    |
| XLM-Roberta-base-CNN-LSTM    | -                  | 99.17     | 99.17  | 99.17    |

#### Tập dữ liệu tiếng Anh (%)

| Model                         | Feature            | Precision | Recall | F1-Score |
| ----------------------------- | ------------------ | --------- | ------ | -------- |
| Logistic Regression           | TF-IDF             | 98.13     | 98.12  | 98.12    |
| Random Forest                 | TF-IDF             | 98.57     | 98.57  | 98.57    |
| XGBoost                       | TF-IDF             | 97.95     | 97.94  | 97.93    |
| Naive Bayes                   | TF-IDF             | 96.23     | 96.18  | 96.18    |
| Extra Trees                   | TF-IDF             | 98.78     | 98.78  | 98.78    |
| LightGBM                      | TF-IDF             | 98.09     | 98.08  | 98.08    |
| Voting Classifier             | TF-IDF             | 98.73     | 98.73  | 98.73    |
| CNN-LSTM Attention (Glove)    | Glove Embedding    | 98.61     | 98.61  | 98.61    |
| CNN-LSTM Attention (Word2Vec) | Word2Vec Embedding | 99.16     | 99.16  | 99.16    |
| DistillBERT-CNN-LSTM          | -                  | 99.52     | 99.52  | 99.52    |
| Bert Base Cased               | -                  | 99.64     | 99.64  | 99.64    |
| XLMRoBerta-base-CNN-LSTM      | -                  | 99.84     | 99.84  | 99.84    |

## Sử Dụng

### Chạy Ứng Dụng

```bash
streamlit run app.py
```
