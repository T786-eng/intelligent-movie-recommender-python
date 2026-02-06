# 🎬 Intelligent Movie Recommendation System

## 📌 Overview
An interactive Movie Recommendation System built with Python. It suggests movies based on content similarity (genres) using **Machine Learning**. 

The system includes a **Smart Search** feature that handles typos and finds the closest matching movie title, making it easy for users to find what they are looking for.

## ✨ Key Features
* **Content-Based Filtering:** Recommends movies similar to the one you like using Cosine Similarity.
* **Interactive CLI:** Runs in a loop, allowing users to keep asking for recommendations.
* **Smart Fuzzy Search:** Handles typos! If you type "Avengr", it correctly identifies "The Avengers".
* **Instant Results:** Processes recommendations in milliseconds.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`
* **Algorithm:** Cosine Similarity on CountVectorized genres.
* **String Matching:** `difflib` (for fuzzy matching).

## 🚀 How to Run
### 1. Install Dependencies
Open your terminal (VS Code: `Ctrl` + `` ` ``) and run:

```bash
pip install pandas numpy scikit-learn
```

Execute the script:
```bash
python movie_recommender.py
```
