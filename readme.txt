# Voice Similarity Search System

Hệ thống tìm kiếm giọng nói phụ nữ tương đồng sử dụng:

* Resemblyzer Speaker Embedding
* MongoDB
* FAISS
* Streamlit UI

---

# 1. Yêu cầu hệ thống

## Python

Khuyến nghị:

```bash
Python 3.10
```

---

# 2. Cài đặt môi trường

## Tạo virtual environment

```bash
py -3.10 -m venv .venv
```

---

## Activate môi trường

### Windows PowerShell

```bash
.venv\Scripts\Activate.ps1
```

---

# 3. Cài dependencies

```bash
pip install numpy==1.26.4
pip install librosa
pip install soundfile
pip install pymongo
pip install faiss-cpu
pip install resemblyzer
pip install streamlit
pip install pandas
```

---

# 4. Cấu trúc project

```text
DPT/

├── dataset/
│   ├── *.wav

├── processed/
│   ├── *.wav

├── indexes/
│
├── src/
│   ├── preprocess.py
│   ├── feature_extractor.py
│   ├── mongodb_store.py
│   ├── build_dataset.py
│   ├── build_index.py
│   ├── search.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 5. Chuẩn bị dataset

Copy các file audio nữ vào:

```text
dataset/
```

Khuyến nghị:

* WAV format
* mono
* ít noise
* 3–10 giây

---

# 6. Khởi động MongoDB

## Local MongoDB

Mở MongoDB service hoặc chạy:

```bash
mongod
```

---

## MongoDB Compass

Database sử dụng:

```text
voice_search
```

Collection:

```text
voices
```

---

# 7. Build dataset features

Lệnh này sẽ:

* preprocess audio
* extract features
* tạo speaker embeddings
* lưu MongoDB

```bash
python src/build_dataset.py
```

Sau khi hoàn thành:

* MongoDB sẽ chứa metadata + feature vectors

---

# 8. Build FAISS index

Lệnh:

```bash
python src/build_index.py
```

Kết quả:

* tạo FAISS vector index
* lưu tại:

```text
indexes/voice.index
```

---

# 9. Test bằng command line

Đặt file test:

```text
query.wav
```

ở root project.

Chạy:

```bash
python main.py
```

Kết quả:

* top 5 audio giống nhất
* similarity score

---

# 10. Chạy giao diện Streamlit

```bash
streamlit run app.py
```

Mở browser:

```text
http://localhost:8501
```

---

# 11. Chức năng của UI

UI hỗ trợ:

* upload audio
* phát audio query
* hiển thị top-5 kết quả
* cosine similarity
* progress bar
* embedding visualization
* feature visualization
* audio playback

---

# 12. Bộ đặc trưng sử dụng

## MFCC

* biểu diễn phổ âm
* đặc trưng vocal tract

---

## Pitch (F0)

* cao độ giọng nói
* hỗ trợ phân biệt người nói

---

## Spectral Centroid

* độ sáng của phổ âm
* texture giọng nói

---

## Speaker Embedding

* vector định danh giọng nói
* trích xuất bằng Resemblyzer

---

# 13. Similarity Search

Hệ thống sử dụng:

```text
Cosine Similarity
```

để đo:

* độ giống giữa các speaker embeddings

---

# 14. Kiến trúc hệ thống

```text
Input Audio
↓
Preprocessing
↓
Feature Extraction
↓
Speaker Embedding
↓
FAISS Similarity Search
↓
Top-5 Retrieval
↓
Streamlit UI
```

---

# 15. Công nghệ sử dụng

| Thành phần        | Công nghệ   |
| ----------------- | ----------- |
| Audio Processing  | librosa     |
| Speaker Embedding | Resemblyzer |
| Metadata Database | MongoDB     |
| Vector Search     | FAISS       |
| User Interface    | Streamlit   |

---

# 16. Một số lưu ý

## Nếu lỗi MongoDB

Kiểm tra MongoDB đã chạy chưa.

---

## Nếu lỗi import package

Cài lại dependencies:

```bash
pip install -r requirements.txt
```

---

## Nếu similarity thấp

Kiểm tra:

* audio noise
* duration quá ngắn
* volume quá nhỏ

---

# 17. Kết quả đầu ra

Hệ thống trả về:

* 5 file giống nhất
* xếp theo similarity giảm dần

Ví dụ:

```text
1.wav → 0.923
18.wav → 0.887
31.wav → 0.862
```

---

# 18. Khả năng mở rộng

Có thể mở rộng:

* real-time microphone
* PCA/t-SNE visualization
* speaker clustering
* multilingual voice search
* large-scale FAISS indexing

---
