# MASCA - Multi-Agent System for Credit Assessment

## Hệ thống đánh giá tín dụng đa tác tử sử dụng Gemini 2.5 Flash-Lite

Hệ thống gồm **3 layer** với **9 agents** chuyên biệt, xử lý song song trong mỗi layer.

---

## 📋 Yêu cầu

- **Python** >= 3.10
- **Gemini API Key** (lấy tại [Google AI Studio](https://aistudio.google.com/apikey))

---

## 🚀 Hướng dẫn cài đặt

### 1. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình API Key

Copy file `.env.example` thành `.env` và điền API key:

```bash
copy .env.example .env
```

Mở file `.env` và sửa:

```env
GEMINI_API_KEY=your-actual-api-key-here
```

### 4. Cấu hình model (tùy chọn)

Các tham số trong `.env` có thể tùy chỉnh:

| Biến | Mặc định | Mô tả |
|---|---|---|
| `GEMINI_MODEL_NAME` | `gemini-2.5-flash-lite` | Tên model Gemini |
| `GEMINI_TEMPERATURE` | `0.3` | Độ sáng tạo (0-1) |
| `GEMINI_MAX_OUTPUT_TOKENS` | `4096` | Số token tối đa output |
| `GEMINI_TOP_P` | `0.95` | Nucleus sampling |
| `GEMINI_TOP_K` | `40` | Top-K sampling |

---

## ▶️ Cách chạy

### Đánh giá 1 sample (mặc định sample 0)

```bash
python main.py
```

### Chọn sample cụ thể (0-999)

```bash
python main.py --sample 42
```

### Bật chế độ debug (xem chi tiết)

```bash
python main.py --sample 0 --verbose
```

### Lưu kết quả ra file JSON

```bash
python main.py --sample 0 --output results.json
```

### Kết hợp tất cả options

```bash
python main.py --sample 10 --verbose --output output_sample_10.json
```

---

## 📊 Đánh giá hàng loạt (Batch Evaluation)

### Chạy đánh giá 10 samples đầu

```bash
python evaluate.py
```

### Chạy 200 test samples (như trong paper)

```bash
python evaluate.py --start 0 --end 200 --output eval_200.json
```

### 💾 Checkpoint (tự động)

Kết quả **tự động lưu sau mỗi sample** vào thư mục `checkpoints/`. Nếu bị gián đoạn (tắt máy, mất mạng, Ctrl+C), chỉ cần **chạy lại cùng lệnh** — hệ thống sẽ tự resume:

```bash
# Lần 1: chạy được 30/200 rồi bị gián đoạn
python evaluate.py --start 0 --end 200

# Lần 2: tự động resume từ sample 31
python evaluate.py --start 0 --end 200
```

### 🔄 Chạy lại từ đầu (bỏ qua checkpoint)

```bash
python evaluate.py --start 0 --end 200 --fresh
```

Kết quả hiển thị:
- ✅ **Accuracy** tổng thể
- 📋 **Confusion Matrix** (TP, FP, TN, FN)
- 📈 **Precision / Recall / F1-Score** cho mỗi class (good/bad)
- ⏱️ **Thời gian trung bình** mỗi sample

---

## 🏗️ Kiến trúc hệ thống

```
Layer 1: Data Ingestion & Contextualization (PARALLEL)
├── Data Analyst         → Chuẩn hóa dữ liệu thô
├── Contextualizer       → Xây dựng persona ứng viên
└── Feature Engineer     → Tính toán features tài chính (DTI, DAR, ...)
        │
        ▼
Layer 2: Multidimensional Assessment (PARALLEL)
├── Risk Modeler         → Phân tích rủi ro tín dụng
├── Income Analyst       → Đánh giá ổn định thu nhập
├── Debt Analyst         → Phân tích nợ & khả năng trả
└── Reward Modeler       → Đánh giá tiềm năng lợi nhuận
        │
        ▼
Layer 3: Strategic Optimization (SEQUENTIAL)
├── Risk-Reward Optimizer → Tối ưu hóa risk/reward
└── Decision Orchestrator → Quyết định cuối: APPROVE / REJECT
```

---

## 📁 Cấu trúc project

```
swinburn_hackathon/
├── config/
│   └── settings.py          # Cấu hình Gemini API (từ .env)
├── data/
│   ├── attribute_map.py     # Mapping 20 attributes German Credit
│   └── loader.py            # Load & parse german.data
├── agents/
│   ├── base.py              # Base agent (Gemini API call)
│   ├── layer1/              # 3 agents Layer 1
│   ├── layer2/              # 4 agents Layer 2
│   └── layer3/              # 2 agents Layer 3
├── prompts/
│   └── templates.py         # Prompt templates (từ paper)
├── pipeline/
│   └── orchestrator.py      # Pipeline orchestrator
├── german_credit_dataset/   # UCI German Credit Dataset
├── main.py                  # Entry point
├── requirements.txt
├── .env.example
└── SETUP.md                 # File này
```

---

## 📊 Dataset

- **German Credit Dataset** (UCI ML Repository)
- 1000 samples, 20 attributes (13 categorical, 7 numerical)
- Label: 1 = Good Credit, 2 = Bad Credit

---

## ⚙️ Đổi model

Chỉ cần sửa 1 dòng trong `.env`:

```env
GEMINI_MODEL_NAME=gemini-2.0-flash
```

Không cần sửa code. Hệ thống hỗ trợ bất kỳ model nào từ Google Gemini API.
