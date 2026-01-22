# 🎬 End-to-End Sentiment Analysis with PyTorch & LSTM

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

## 🚀 About the Project & My Deep Learning Journey

As a Data Scientist, I have advanced my journey in Artificial Intelligence from Classical Neural Networks (ANN) and Convolutional Neural Networks (CNN) to the analysis of **Sequential Data**.

This project aims to analyze context and sentiment in textual data (NLP). It is built upon the **LSTM (Long Short-Term Memory)** architecture, which overcomes the *Vanishing Gradient* problem often encountered in "Vanilla RNNs."

The development process focuses on going beyond pre-built libraries to deeply understand and optimize the mathematical concepts behind **Gradient Clipping**, **Hidden State Management**, and **Embedding Layers**.

## 🔄 Project Evolution: Two-Phase Approach

To enhance data processing skills and model generalization capability, this project was executed in two distinct phases.

### Phase 1: Kaggle Dataset & Manual Pipeline
In the first phase, the `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews` dataset from **Kaggle** was used.
* **Goal:** To process raw data, apply manual preprocessing steps, and establish the fundamental LSTM structure.
* **Method:** CSV manipulation with Pandas and construction of a custom Tokenizer.

### Phase 2: Hugging Face Integration & Optimization
In the second phase, the **Hugging Face** `imdb` dataset was utilized to align with industry standards.
* **Goal:** To increase data loading speed, create a standardized benchmark, and optimize model architecture.
* **Method:** Streaming data flow and dynamic batching using the `datasets` library.

| Feature | Phase 1 (Kaggle) | Phase 2 (Hugging Face) |
| :--- | :--- | :--- |
| **Data Source** | CSV File (Local/Drive) | Cloud Dataset (API) |
| **Preprocessing** | Manual Pandas Operations | On-the-fly Tokenization |
| **Focus** | Data Manipulation & Base Structure | Performance & Scalability |

## 🏗️ Model Architecture

The model is a custom `nn.Module` class built with PyTorch, tested on both datasets.

| Layer | Description |
| :--- | :--- |
| **Embedding** | Maps words to a 128-dimensional vector space (Domain-specific representation). |
| **LSTM** | 2-Layer (Stacked) with 256 Hidden Dimensions. Preserves "Context" information in sequential data. |
| **Dropout** | Randomly zeros out 30% of neurons to prevent overfitting. |
| **Fully Connected** | Reduces LSTM output to a single score. |
| **Sigmoid** | Squeezes the output between 0 and 1 to produce a probability. |

## 💡 Technical Deep Dive (Key Takeaways)

Advanced techniques applied during the project:

1.  **LSTM Memory Cells:**
    Unlike standard RNNs, the "Forget Gate" mechanism was utilized to preserve context in long sentences.

2.  **Gradient Clipping:**
    To ensure training stability and prevent the "Exploding Gradient" risk, gradient norms were capped (`clip=5`).

3.  **Hidden State Management:**
    A mechanism to reset the Hidden State at each iteration was implemented to prevent memory leakage during batch training.

4.  **Learnable Embeddings:**
    Instead of using pre-trained vectors, the model was designed to learn word relationships specific to the IMDB dataset from scratch (Dynamic Embedding).

## 📊 Results

The model was trained on an **L4 GPU**, yielding the following metrics:

- **Train Accuracy:** ~87%
- **Validation Accuracy:** ~83%
- **Inference:** The model successfully detects irony and contextual sentiment shifts in unseen data.

## 🛠️ Installation & Usage

```bash
git clone [https://github.com/ccemozclk/Sentiment-LSTM-PyTorch.git](https://github.com/ccemozclk/Sentiment-LSTM-PyTorch.git)
pip install torch datasets numpy pandas
