# Natural Language Processing with Disaster Tweets

**Author:** C. McGinnis  
**Date:** November 27, 2025  
**Course:** Machine Learning Mini-Project  
**Competition:** [Kaggle – Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

---

## 1. Project Overview

In the age of social media, Twitter has become a critical real-time communication channel during emergency situations. Millions of tweets are posted every day, and many of them contain disaster-related keywords like *fire*, *flood*, *earthquake*, or *explosion*. However, not all of these tweets describe actual emergencies. Tweets can be metaphorical (“my timeline is on fire”), humorous, or sarcastic.

This mini-project explores how to automatically distinguish tweets that report **real disasters** from those that do **not**. Using Recurrent Neural Network (RNN) architectures, I build and evaluate text classification models that operate directly on tweet text.

The work was implemented in **Google Colab** using **TensorFlow/Keras** and follows a full machine learning workflow:

- Data exploration  
- Text preprocessing  
- RNN-based model development (Vanilla RNN, LSTM, GRU)  
- Hyperparameter tuning (small manual search)  
- Model comparison and selection  

The final selected model is a **GRU-based classifier** that achieves an F1 score of approximately **0.77** on the validation set.

---

## 2. Dataset

This project uses the labeled dataset from the Kaggle competition:

> **Natural Language Processing with Disaster Tweets**  
> https://www.kaggle.com/c/nlp-getting-started

Each tweet is annotated as:

- `1` – The tweet is about a **real disaster**  
- `0` – The tweet is **not** about a real disaster  

The dataset also includes optional `keyword` and `location` fields, but the core models in this project focus primarily on the tweet text.

> **Note:** The full dataset is not included in this repository due to Kaggle’s terms of use.  
> Please download it directly from Kaggle and place it in a `data/` directory, or update the paths in the notebook accordingly.

---

## 3. Methods

### 3.1 Preprocessing

Main preprocessing steps:

- Lowercasing tweet text  
- Removing URLs, user mentions, and some punctuation  
- Tokenization using Keras’ `Tokenizer`  
- Converting tokens to integer sequences  
- Padding sequences to a fixed maximum length (`max_len`)  

The tokenized and padded sequences (`X_train`, `X_val`, etc.) are used as inputs to all models.

### 3.2 Models

I experimented with three recurrent architectures:

1. **Vanilla RNN**  
2. **LSTM (Long Short-Term Memory)**  
3. **GRU (Gated Recurrent Unit)**  

All models share a common structure:

- **Embedding layer** (e.g., 100-dimensional embeddings)  
- **Recurrent layer** (RNN/LSTM/GRU with 64 units)  
- **Dropout** for regularization  
- **Dense layer** with ReLU activation  
- **Output layer** with a single sigmoid unit for binary classification  

Training details:

- Loss: `binary_crossentropy`  
- Optimizer: `Adam` (learning rate = 0.001)  
- Batch size: 32  
- Early stopping and learning rate reduction based on validation loss  

### 3.3 Hyperparameter Tuning (Section 4.4 in report)

I performed a small, manual hyperparameter search over four configurations:

- `GRU_dropout_0.3` – GRU with dropout = 0.3  
- `GRU_dropout_0.5` – GRU with dropout = 0.5  
- `LSTM_dropout_0.5` – LSTM with dropout = 0.5  
- `RNN_dropout_0.5` – Vanilla RNN with dropout = 0.5  

All used:

- Embedding dimension: 100  
- Recurrent units: 64  
- Dense units: 32  
- Learning rate: 0.001  
- Batch size: 32  
- Max epochs: 20, with early stopping  

Validation performance (sorted by F1):

- **GRU (dropout 0.5)** – F1 = **0.7638**, val loss ≈ 0.478, accuracy ≈ 0.789  
- **GRU (dropout 0.3)** – F1 = 0.7576, val loss ≈ 0.475, accuracy ≈ 0.794  
- **LSTM (dropout 0.5)** – F1 = 0.7057, val loss ≈ 0.472, accuracy ≈ 0.785  
- **RNN (dropout 0.5)** – F1 = 0.6672, val loss ≈ 0.548, accuracy ≈ 0.739  

The GRU models clearly outperform the LSTM and Vanilla RNN on F1.

---

## 4. Final Model: GRU

The final selected model is a GRU with:

- Embedding dimension: 100  
- GRU units: 64  
- Dropout: 0.5  
- Dense units: 32  

Final validation performance:

- **Weighted Score:** 0.7731 (internal composite metric)  
- **Accuracy:** 0.8056 (80.56%)  
- **F1 Score:** 0.7734  
- **Precision:** 0.7745  
- **Recall:** 0.7722  
- **Total Parameters:** 1,034,145  
- **Epochs Trained:** 10  
- **Best Epoch:** 5  
- **Best Validation Loss:** 0.4430  

The F1 score around 0.77 with balanced precision and recall suggests the model is a reasonable first-pass filter for identifying disaster-related tweets.

![GRU Model Architecture](https://github.com/user-attachments/assets/7c8170e2-28db-4d3e-a9f8-4c5c9a6e2e33)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PPVuIhWD9gtE8jfWusmGPdKWn5MpVS2m?usp=sharing)

```markdown

