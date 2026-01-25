# 🎬 End-to-End Sentiment Analysis with Bidirectional LSTM & PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 🚀 About the Project & My Deep Learning Journey

As a Data Scientist, I have advanced my journey in Artificial Intelligence from Classical Neural Networks (ANN) and Convolutional Neural Networks (CNN) to the complex world of **Sequential Data**.

This project is a deep dive into **Natural Language Processing (NLP)**, aiming to detect sentiment (Positive/Negative) in movie reviews. While it started with a standard LSTM architecture, it evolved into a robust **Bidirectional LSTM (Bi-LSTM)** model to master context in long sequences.

The project goes beyond simple library implementations, focusing on **custom architecture design**, **memory management**, **smart preprocessing**, and **solving the vanishing gradient problem**.

## 🔄 Project Evolution: The "Smart" Pipeline

To ensure robustness and scalability, I implemented a unified pipeline that handles different data sources seamlessly.

### Phase 1: Kaggle Dataset (Manual Pipeline)
Used the `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews` dataset.
* **Challenge:** Processing raw CSV data and handling noise (HTML tags, punctuation).
* **Solution:** Developed a custom `preprocess_and_tokenize` function that standardizes text cleaning.

### Phase 2: Hugging Face Integration (Cloud & Speed)
Utilized the `imdb` dataset via the `datasets` library.
* **Challenge:** Aligning cloud-based data structures with local preprocessing logic.
* **Solution:** Implemented a **"Smart Vocab Builder"** that detects input types (List vs. Dataset) and processes them through a single, unified engine.

## 🏗️ Model Architecture: Bidirectional LSTM

The heart of this project is a custom `nn.Module` class that processes text in **both directions** (past-to-future and future-to-past).

| Layer | Type | Description |
| :--- | :--- | :--- |
| **Embedding** | `nn.Embedding` | Learnable Vector Space (128-dim). uses `padding_idx=0` to ignore padding tokens efficiently. |
| **Bi-LSTM** | `nn.LSTM` | **Bidirectional=True**. 2 Stacked Layers. Captures context from both the beginning and end of sentences. |
| **Concatenation** | Tensor Op | Combines the final hidden states of forward and backward passes (`Hidden x 2`). |
| **Dropout** | `nn.Dropout` | Applied at 50% (`p=0.5`) to enforce generalization and prevent overfitting. |
| **Classifier** | `nn.Linear` | Fully Connected layer mapping the combined context to a sentiment score. |

## 💡 Technical Deep Dive (Key Takeaways)

1.  **Why Bidirectional?**
    Standard LSTMs often "forget" the beginning of a long paragraph by the time they reach the end. Bi-LSTM reads the review backwards as well, preserving crucial initial context (e.g., "I *hated* this movie...").

2.  **Smart Preprocessing & Padding:**
    A unified function handles HTML tag removal (`<br />`) and ensures consistent tokenization across Training, Validation, and Inference phases. `padding_idx=0` was implemented to prevent the model from learning "noise" from empty padding slots.

3.  **Gradient Clipping:**
    To ensure training stability and prevent the "Exploding Gradient" risk, gradient norms were capped (`clip=5`).

4.  **Dynamic Learning:**
    No pre-trained embeddings (like GloVe) were used. The model learned the semantic relationships of the IMDB domain *from scratch*.

## 📊 Results & Performance

The model was trained on an **L4 GPU** with an optimized strategy (learning rate scheduling & early stopping).

| Metric | Kaggle Dataset | Hugging Face Dataset |
| :--- | :--- | :--- |
| **Validation Accuracy** | **88.54%** 🏆 | **88.38%** |
| **Validation Loss** | 0.28 | 0.29 |

### 🧠 Edge Case Analysis (Inference)
The model successfully handles tricky linguistic structures that confuse simpler models:

* ✅ **Irony:** *"Best movie ever? I don't think so."* → **Detected as Negative**
* ✅ **Double Negatives:** *"It was not bad at all."* → **Detected as Positive**
* ✅ **Contextual Shift:** *"I really wanted to like this movie but I couldn't."* → **Detected as Negative**

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ccemozclk/Sentiment-LSTM-PyTorch.git](https://github.com/ccemozclk/Sentiment-LSTM-PyTorch.git)
