# 🎬 Netflix Recommendation System

A **content-based machine learning recommendation system** built on the Netflix Titles dataset. Given a title, genre, or mood description, the system recommends the most relevant Netflix movies and TV shows using TF-IDF vectorization and multiple similarity models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Models & Evaluation](#models--evaluation)
- [Visualizations](#visualizations)
- [Example Outputs](#example-outputs)
- [Technologies Used](#technologies-used)

---

## Overview

This project builds an end-to-end recommendation system that:

- Cleans and preprocesses raw Netflix catalog data
- Converts categorical and text features into numeric representations
- Trains and compares three recommendation models
- Selects the best-performing model (SVD Latent Factor — **99.2% Precision@10**)
- Provides an interactive CLI to get recommendations by title, genre, or mood

---

## Dataset

**Source:** [Netflix Movies and TV Shows – Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)

| Property | Value |
|---|---|
| File | `netflix_titles.csv` |
| Rows | 8,807 titles |
| Columns | 12 |
| Content | Movies & TV Shows on Netflix |

**Key columns used:** `title`, `type`, `director`, `cast`, `country`, `listed_in`, `description`, `release_year`, `rating`, `duration`

---

## Project Structure

```
netflix-recommendation-system/
│
├── netflix_titles.csv                  # Dataset (download from Kaggle)
├── netflix_recommendation_system.py    # Main ML script
├── README.md                           # This file
│
└── outputs/
    ├── netflix_eda.png                 # Exploratory data analysis charts
    ├── netflix_correlation.png         # Feature correlation heatmap
    └── netflix_model_comparison.png    # Model evaluation bar chart
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/netflix-recommendation-system.git
cd netflix-recommendation-system
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 3. Download the dataset

Download `netflix_titles.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows) and place it in the project root directory.

---

## Usage

Run the script from your terminal:

```bash
python3 netflix_recommendation_system.py
```

You will see an interactive menu with 4 modes:

```
======================================================================
   🍿  NETFLIX RECOMMENDATION SYSTEM  🍿
======================================================================
  Mode 1 : Recommend by Title
  Mode 2 : Recommend by Genre
  Mode 3 : Recommend by Description / Mood
  Mode 4 : Quick Demo (auto-run examples)
  quit   : Exit
======================================================================
```

### Mode 1 — Recommend by Title

Find titles similar to one you already like:

```
Enter mode: 1
Enter a Netflix title: Stranger Things
How many recommendations? 10
```

### Mode 2 — Recommend by Genre

Browse top titles in a specific genre:

```
Enter mode: 2
Enter a genre: Horror Movies
Filter by type? (Movie / TV Show / leave blank): Movie
How many recommendations? 10
```

### Mode 3 — Recommend by Mood / Description

Describe what you feel like watching in plain English:

```
Enter mode: 3
Describe what you want to watch: romantic comedy with funny moments and a happy ending
How many recommendations? 10
```

### Mode 4 — Quick Demo

Runs 3 pre-set examples automatically to showcase all modes.

---

## Pipeline

```
Raw CSV Data
     │
     ▼
Data Cleaning
  • Fill missing values (director, cast, country, rating, duration)
  • Remove duplicate titles
  • Parse date_added → year_added
  • Extract duration_mins / duration_seasons
     │
     ▼
Type Conversion
  • LabelEncoder  → type, rating (category → numeric)
  • MultiLabelBinarizer → genres (42 binary columns)
     │
     ▼
Feature Engineering
  • Build text "soup": genres + director + top-4 cast + type + description
  • TF-IDF Vectorizer (15,000 features, bigrams)
     │
     ▼
Model Training
  • Model A: TF-IDF + Cosine Similarity
  • Model B: TF-IDF + KNN (k=11, cosine)
  • Model C: TF-IDF + TruncatedSVD (100 latent factors)
     │
     ▼
Evaluation (Precision@10 on 50 random samples)
     │
     ▼
Best Model Selected → Interactive Recommendation CLI
```

---

## Models & Evaluation

All three models were evaluated using **Precision@10** — the fraction of the top 10 recommendations that share at least one genre with the query title.

| Model | Precision@10 |
|---|---|
| TF-IDF Cosine Similarity | 0.864 |
| KNN (cosine distance) | 0.864 |
| **TruncatedSVD Latent Factor** ✅ | **0.992** |

**Winner: TruncatedSVD** with 99.2% precision, leveraging 100 latent semantic dimensions to capture deeper relationships between content features.

---

## Visualizations

The script automatically generates and saves three charts:

| Chart | Description |
|---|---|
| `netflix_eda.png` | 6-panel EDA: type split, top countries, yearly growth, top genres, rating distribution, movie duration histogram |
| `netflix_correlation.png` | Heatmap of correlations between numeric features |
| `netflix_model_comparison.png` | Bar chart comparing Precision@10 across all three models |

---

## Example Outputs

**Mode 1 — Similar to "Stranger Things":**

```
 1. [TV Show]  Dark  (2017)
    Genre  : Crime TV Shows, International TV Shows, TV Mysteries, TV Sci-Fi & Fantasy
    Rating : TV-MA  |  Similarity: 0.5821

 2. [TV Show]  The OA  (2019)
    Genre  : TV Dramas, TV Mysteries, TV Sci-Fi & Fantasy
    Rating : TV-MA  |  Similarity: 0.4903
```

**Mode 3 — Mood: "crime thriller with suspense and a twist ending":**

```
 1. [TV Show]  Clickbait  (2021)
    Genre  : Crime TV Shows, TV Dramas, TV Mysteries
    Rating : TV-MA  |  Similarity: 0.1859

 2. [TV Show]  Longmire  (2017)
    Genre  : Crime TV Shows, TV Dramas
    Rating : TV-MA  |  Similarity: 0.1724
```

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | TF-IDF, TruncatedSVD, KNN, LabelEncoder, cosine similarity |
| `scipy` | Sparse matrix handling |

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgements

- Dataset provided by [Shivam Bansal on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- Netflix branding and titles are property of Netflix, Inc.

