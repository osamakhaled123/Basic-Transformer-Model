# Basic Transformer Model
# Transformer from Scratch for Abstractive Summarization

This project implements a Transformer-based encoder-decoder model **from scratch** using PyTorch, designed for **abstractive text summarization** on the [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail). It includes a complete pipeline for preprocessing, model training, beam search inference, and BLEU score evaluation.

---

## 📌 Key Features

- 🔧 Implemented the Transformer architecture from scratch (no HuggingFace or pretrained models).
- ✂️ Cleaned and tokenized dataset manually, including:
  - Lowercasing
  - Removing punctuation and numbers
  - Filtering short samples
  - Padding/truncating to fixed lengths
- 🧠 Beam search decoder with repetition handling
- 📉 Training and validation loss tracking
- 📈 Final **BLEU score: 22**
- 📝 Python implementation + interactive training notebook

---

## 📊 Results

| Metric        | Value          |
|---------------|----------------|
| Training Loss | 24.429 → 8.371 |
| Validation Loss | 9.386 → 8.371 |
| BLEU Score    | 22             |
| Epochs        | 5              |
| Batch Size    | 32             |
| Dropout       | 0.5            |

---

## 🧱 Project Structure
.
├── Processing_Summarizing_Datasets_From_Scratch.py # Custom text cleaning & tokenization
├── Transformer_model.py # Transformer encoder/decoder classes
├── summerizing.ipynb # Interactive training & testing notebook
└── README.md

📈 Example Loss Curve
![image](https://github.com/user-attachments/assets/311f22b4-70ac-4348-99aa-ec569bafb457)


🤝 Acknowledgments

    PyTorch

    Hugging Face Datasets

    CNN/DailyMail Dataset




