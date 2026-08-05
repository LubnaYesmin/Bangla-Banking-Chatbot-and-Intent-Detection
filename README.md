# 🇧🇩 Bangla Banking Chatbot and Intent Detection

An advanced Natural Language Processing (NLP) framework designed for intent classification in the banking sector. This repository benchmarks **five distinct machine learning and deep learning architectures** to detect user intents from code-mixed (Banglish) and native Bengali script queries.

---

## 🚀 System Architecture Overview

The system processes multi-script user queries (e.g., *"amar balance koto"* or *"আমার এটিএম কোথায়?"*) and routes them into 10 target banking categories using parallel modeling strategies:

---

## 📊 Dataset Specifications & Features

- **Initial Dataset Volume:** 10,000 conversational instances.
- **Pre-processing:** Programmatic deduplication reduced the final core matrix to **533 unique, high-variance semantic samples** to eliminate data leakage.
- **Intents Covered (10 Classes):** 
  `account_open`, `atm_location`, `balance_check`, `card_issue`, `cheque_book`, `customer_support`, `loan_query`, `mobile_banking`, `money_transfer`, `transaction_history`.

---

## ⚡ Model Profiles & Hyperparameters

| Model Configuration | Tokenizer Strategy | Key Hyperparameters | Target Strengths |
| :--- | :--- | :--- | :--- |
| **MuRIL BERT** | WordPiece Subword | Epochs: 3 \| Batch Size: 8 \| Max Length: 64 | Superior cross-lingual mapping for phonetic/Banglish strings. |
| **BanglaBERT** | SentencePiece Monolingual | Epochs: 5 \| Learning Rate: 2e-5 \| Batch: 8 | Deep syntactic understanding of native Bengali script grammar. |
| **mBERT** | Multilingual WordPiece | Epochs: 5 \| Learning Rate: 2e-5 \| Batch: 8 | High generalization capability across multi-language contexts. |
| **BiLSTM + FastText** | Custom Word-Level Vocabulary | Epochs: 5 \| Embedding Dim: 300 \| Hidden Dim: 128 | Extremely low latency; exceptional performance on filtered datasets. |
| **Logistic Regression** | TF-IDF (Unigram/Bigram) | Max Iterations: 1000 \| Max Features: 5000 | Highly resource-efficient baseline engine. |

---

## 📈 Empirical Benchmarking Results

The systems were benchmarked against test sets under split metrics (80/20 for MuRIL/LR and 70/15/15 for Deep Learning models):

| Model Architecture | Testing Accuracy | Training Loss (Final) | Validation/Test Loss |
| :--- | :---: | :---: | :---: |
| **MuRIL BERT** | **100.00%** | 0.2261 | 0.0031 |
| **BiLSTM + FastText** | **100.00%** | 0.0108 | 0.0143 |
| **TF-IDF + Logistic Regression** | **96.26%** | 0.4810 | 0.6431 |
| **BanglaBERT** | **92.50%** | 0.6838 | 0.6372 |
| **mBERT** | **92.50%** | 0.1696 | 0.4216 |

> **Note on Performance:** MuRIL BERT demonstrated complete convergence due to its specialized pre-training on Indic language variations. The Logistic Regression framework was intentionally challenged with a 4-sample label noise injection to evaluate system robustness.

---

## 🛠️ Execution & Installation Guide

### Dependencies Setup
```bash
pip install transformers datasets torch scikit-learn pandas openpyxl matplotlib seaborn tqdm
```

### Quick Inference Routine
To run manual validation using the highly optimized MuRIL pipeline:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the trained configurations
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("./your_saved_model_path")

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return prediction.item()

print(predict_intent("amar account a koto taka ache ekhon")) # Target Output: balance_check
```
