# PhoBERT Medical NER System

**Free, self-hosted Vietnamese medical NER** - Extract diseases (Bệnh), symptoms (Triệu chứng), and medications (Thuốc) from Vietnamese medical text.

---

## 🎯 What This System Does

Replaces your **Gemini-based NER pipeline** with a **free, self-hosted PhoBERT model**:

| Feature | Gemini (Current) | PhoBERT (New) |
|---------|------------------|---------------|
| **Cost** | 💰 Paid API | 🆓 Free |
| **Speed** | 🐌 0.5 chunks/sec | ⚡ 10-50 chunks/sec |
| **Internet** | 🌐 Required | 📡 Offline |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Categories** | 8 (all) | 3 (diseases, symptoms, drugs) |

---

## 📁 Files Overview

### Configuration
- **`phobert_config.py`** - All settings (paths, hyperparameters)
- **`requirements_phobert.txt`** - Python dependencies

### Core Scripts
- **`phobert_data_processor.py`** - Load & preprocess ViMedNER BIO data
- **`phobert_train.py`** - Fine-tune PhoBERT on your data
- **`phobert_inference.py`** - Extract entities from text
- **`phobert_evaluate.py`** - Test model performance
- **`phobert_ner_pipeline.py`** - Production pipeline (replaces NER_extractor.py)

### Helper Scripts
- **`quick_start.py`** - Check setup and guide you
- **`PHOBERT_SETUP_GUIDE.md`** - Detailed documentation
- **`PHOBERT_README.md`** - This file

---

## 🚀 Quick Start (3 Commands)

### 1. Install Dependencies
```bash
cd "NER LLM"
pip install -r requirements_phobert.txt
```

### 2. Setup Data & Config
- Place ViMedNER data in `vimedner_data/`
- Update paths in `phobert_config.py`

### 3. Train Model
```bash
python phobert_train.py
```

Done! Now use it:
```bash
python phobert_ner_pipeline.py
```

---

## 📊 Output Format

```json
[
  {
    "content": "Sỏi thận gây đau quặn thận và đái máu...",
    "entities": {
      "Bệnh": ["Sỏi thận"],
      "Triệu chứng": ["đau quặn thận", "đái máu"],
      "Thuốc": []
    }
  }
]
```
