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

### Prerequisites
* Python 3.8 or higher
* CUDA-compatible GPU recommended (CPU also supported, but training will be significantly slower)
* ~2GB free disk space for datasets and model checkpoints

### 1. Clone the Repository

```bash
git clone https://github.com/ccemozclk/Sentiment-LSTM-PyTorch.git
cd Sentiment-LSTM-PyTorch
```

### 2. Set Up the Environment

It is recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib
pip install datasets transformers
pip install kaggle
```

Alternatively, if a `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

You can choose between two data sources:

**Option A — Hugging Face (recommended for quick start):**
The dataset will be auto-downloaded by the `datasets` library on first run. No manual setup needed.

**Option B — Kaggle:**
1. Download `IMDB Dataset.csv` from [Kaggle IMDB Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Place it under `data/` directory in the project root.

### 5. Train the Model

```bash
python train.py
```

The script will:
* Load and preprocess the data
* Build the vocabulary
* Initialize the Bi-LSTM model
* Train with early stopping (default: 10 epochs, patience=3)
* Save the best model to `models/best_model.pth`

### 6. Run Inference on Custom Text

```bash
python predict.py --text "This movie was an absolute masterpiece."
```

Or programmatically:

```python
from predict import load_model, predict_sentiment

model, vocab = load_model('models/best_model.pth')
sentiment, confidence = predict_sentiment(
    "It was not bad at all.",
    model,
    vocab
)
print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")
```

## 📁 Project Structure

```
Sentiment-LSTM-PyTorch/
├── data/                          # Dataset files (Kaggle CSV, etc.)
├── models/                        # Saved model checkpoints
│   └── best_model.pth
├── notebooks/                     # Jupyter exploration notebooks
│   └── 01_EDA_and_Training.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Smart Vocab Builder + tokenizer
│   ├── model.py                   # Bidirectional LSTM architecture
│   ├── trainer.py                 # Training loop with early stopping
│   └── utils.py                   # Helper functions
├── train.py                       # Main training entry point
├── predict.py                     # Inference script
├── requirements.txt
└── README.md
```

## 🎓 Key Learnings

This project pushed me through several technical and conceptual milestones:

* **Sequential Data Mindset:** Moved beyond independent feature thinking (typical in tabular ML) to capturing temporal/positional dependencies in text.
* **Custom Architecture Design:** Built a bidirectional model from scratch rather than relying on `transformers` library shortcuts — gaining intuition about hidden state flow.
* **Production Readiness:** Implemented gradient clipping, learning rate scheduling, and early stopping — standard practices in production ML systems.
* **Domain Adaptation:** Trained embeddings from scratch on IMDB rather than using GloVe, demonstrating that domain-specific learning can match or exceed transfer learning for narrow corpora.

## 🚀 Future Improvements

Potential extensions to push this project further:

* [ ] Replace LSTM with **Transformer encoder** for parallelizable training
* [ ] Add **attention mechanism** for interpretable sentiment scoring
* [ ] Fine-tune **BERT** as a benchmark comparison
* [ ] Deploy as a **FastAPI** REST endpoint
* [ ] Add **ONNX export** for cross-platform inference
* [ ] Extend to **multi-class sentiment** (very negative → very positive)

## 📚 References & Inspiration

* Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
* Schuster & Paliwal (1997) — *Bidirectional Recurrent Neural Networks*
* PyTorch Official LSTM Tutorial
* Hugging Face `datasets` documentation

## 📬 Contact

* **Author:** İsmail Cem Özçelik
* **Role:** Data Scientist & Industrial Engineer
* **Email:** i.cemozcelik@gmail.com
* **LinkedIn:** [linkedin.com/in/cemozcelık](https://linkedin.com/in/cemozcelık)
* **GitHub:** [github.com/ccemozclk](https://github.com/ccemozclk)

---

*This project reflects my passion for building deep learning systems from first principles — understanding what happens inside the architecture, not just calling library functions.*

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
