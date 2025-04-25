# Amazon-Product-Pricer
fine tuning a frontier model for predicting the price of items

# 🛍️ Product Pricer

**A machine learning pipeline that estimates product prices based on their descriptions using both classical ML techniques and fine-tuned Large Language Models (LLMs).**

---

## 📌 Project Overview

This project builds a robust **product price prediction system** leveraging a subset of the [Amazon Reviews 2023 Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). It aims to compare traditional machine learning baselines with modern LLMs such as GPT-4o-mini (via OpenAI), including fine-tuned models on curated data.

---

## 🧠 Approach

### 1. 🔍 Data Curation & Preprocessing

- **Dataset Source**: [McAuley Lab's Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories)
- Focused on the **Home Appliances** category to reduce compute cost.
- Filtered items with prices between **$1 and $999**.
- Truncated item descriptions to ~**180 tokens** for efficient training and inference alignment.
- Built structured `Item` objects with prompt-response format for training.

### 2. 📊 Data Split

- **Training Set**: 25,000 items  
- **Test Set**: 2,000 items (subset of 250 used for visual and metric evaluation)

---

### 3. 🧪 Baseline Models

We experimented with several traditional models for comparison:

- 🔢 **Random Number Generator** (control baseline)
- 📈 **Linear Regression** with:
  - Basic handcrafted features
  - **Word2Vec** vector embeddings
- 🧠 **Support Vector Machines** (SVM) using Word2Vec features
- 🌲 **Random Forests** using Word2Vec

---

### 4. 🤖 LLM Predictions

- **GPT-4o-mini** used in zero-shot mode for price estimation.
- Fine-tuned **GPT-4o-mini** using curated JSONL prompt dataset uploaded via the OpenAI API.
- Training tracked using **Weights & Biases**, integrated with OpenAI's fine-tuning flow.

---

## 🛠️ Technologies Used

- 🐍 Python 3.x
- 🤗 HuggingFace Datasets
- 🔡 OpenAI API (fine-tuning + inference)
- 📦 Scikit-learn (for traditional models)
- 🧠 Gensim / Word2Vec
- 📊 Matplotlib
- 🪄 Weights & Biases (wandb)

---

## 🧰 Helper Scripts

### `items.py`

Defines the `Item` class, responsible for:

- Cleaning and sanitizing text fields.
- Truncating tokens for training consistency.
- Constructing training/inference prompts in a structured format.
- Filtering out low-quality data points.

### `loaders.py`

Implements `ItemLoader`, used to:

- Load and process Amazon product data.
- Filter products by price.
- Batch process datapoints in parallel (via `ProcessPoolExecutor`) for speed.
- Create valid `Item` instances for downstream use.

### `testing.py`

Defines the `Tester` class, which:

- Runs model predictions over test data.
- Computes metrics like absolute error and RMSLE.
- Assigns a performance color code to each prediction (green/orange/red).
- Plots predicted vs. actual prices for visual inspection.
- Includes a one-liner API:  
  ```python
  Tester.test(my_prediction_function, test_data)




## 📚 References
-  [Amazon Reviews Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories)
-  [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
-  [Weights & Biases](https://wandb.ai/)
-  Course Reference: Portions of this project (including Item, ItemLoader, and Tester class patterns) are repurposed from the excellent [LLM Engineering course by Ed Donner on Udemy](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/?srsltid=AfmBOorKMuFoTz7AXX1gI45R_weqWMHodU1Fw-aR0E84b3gACOxwqWkS&couponCode=LEARNNOWPLANS)
