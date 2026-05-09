<div align="center">
  
# Technical Evaluation of Women’s Safety Trends Through Large-Scale Twitter Data Mining and Analysis



## AI-Powered Social Media Sentiment Analysis for Women’s Safety Monitoring

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-purple?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-cyan?style=for-the-badge)
![Research](https://img.shields.io/badge/Research-Conference%20Project-black?style=for-the-badge)
![Twitter Data](https://img.shields.io/badge/Dataset-Twitter%20Data-1DA1F2?style=for-the-badge\&logo=x)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

### 📊 Exploring Public Sentiment and Safety Trends Using NLP & Machine Learning

</div>

---

# Overview

Women’s safety remains one of the most critical social concerns worldwide. Traditional methods of understanding safety-related issues are often limited, delayed, and unable to capture real-time public sentiment.

This research project leverages **Natural Language Processing (NLP)** and **Machine Learning** techniques to analyze large-scale Twitter discussions related to women’s safety, harassment, fear, and unsafe experiences.

The system extracts meaningful insights from unstructured textual data and evaluates public sentiment trends using multiple machine learning models.

---

# Problem Statement

* Traditional women’s safety analysis methods are slow and limited.
* Social media platforms generate massive real-time discussions.
* Twitter data reflects fear, harassment, and unsafe public experiences.
* Extracting meaningful insights from unstructured textual data remains a major challenge.

---

# Objectives

* Analyze large-scale Twitter data related to women’s safety discussions.
* Apply NLP and Machine Learning techniques for sentiment classification.
* Identify patterns, trends, and public concerns from social media interactions.
* Evaluate the performance of multiple machine learning models.
* Support data-driven awareness and intelligent safety monitoring systems.

---

# Research Pipeline

<img width="400" height="550" alt="fin drawio" src="https://github.com/user-attachments/assets/872f037b-0b2e-41c4-a316-8bf871e8a9c4" />

---

# Models Used

| Model               | Purpose                         | Performance           |
| ------------------- | ------------------------------- | --------------------- |
| Logistic Regression | Sentiment Classification        | High Accuracy         |
| Naïve Bayes         | Text Probability Classification | Efficient Performance |
| Random Forest       | Ensemble Classification         | Robust Analysis       |
| SVM                 | Complex Text Classification     | Best Overall Accuracy |
| Decision Tree       | Interpretable Classification    | Moderate Accuracy     |

---

# Literature Review

| Ref            | NLP Model   | Sentiment Analysis Focus                       | Pre-Processing Techniques                   |  Key Contributions   |
| -------------- | ----------  | ----------------------------------------       | ---------------------------------------     | ---------------------
| Lagrari et al. | BERT        | General Sentiment Analysis on social media     | Tokenization, Stop Word Removal, Stemming   | Demonstrated robust accuracy for binary classification (Acc.: 92%) 
| Qaisar et al.  | LSTM        | Long-text Sentiment Analysis                   | Data Cleaning, Lemmatization, POS Tagging   | Addressed performance on detailed movie review datasets (F1: 0.89)
| Jahin et al.   | RoBERTa     | Crisis-based Sentiment Analysis                | Noise Removal, Tokenization, Case Folding   | Effective for analyzing public sentiment during the Covid-19 pandemic (Acc.: 94%)
| Kashid et al.  | BİLSTM-CNN  | Product Sentiment Analysis                     | TF-IDF, Data Balancing, Stop Word Removal   | High precision for product review analysis (Precision: 0.87)
| Alipour et al. | GPT-2       | Cross-platform Sentiment Analysis              | Tokenization, Dependency Parsing            | Showed adaptability to multiple social media platforms (Acc.: 88%)
| Amini et al.   | Transformer | Analysis for informal text sentiment           | Data Augmentation, Label Encoding           | Highlighted limitations in adapting to informal language (F1: 0.82)
| Sittar et al.  | BERT + LSTM | Sentiment analysis in news media               | Tokenization, Feature Vector Formation      | Achieved high accuracy on structured text analysis (Acc.:95%)
| Qorich et al.  | CNN         | Binary sentiment classification                | Normalization, Lemmatization                | Balanced accuracy for straightforward sentiment tasks (Acc.:90%)
| Nadi et al.    | GPT-3.5     | Real-time sentiment tracking                   | Tokenization, Noise Removal                 | Effective in capturing real-time public reactions (F1: 0.86)
| Jiang et al.   | BERT + CNN  | Political sentiment analysis                   | Case Folding, Removal of Hashtags           | Reliable for sentiment in politically charged discourse (Acc.:96%)
| Proposed System | Hybrid NLP pipeline with RoBERTa and classical models | Analyzes emotional tone of tweets to identify distress, urgency, fear, anger, and safety concerns. | Tokenization, stopword removal, stemming, normalization, hashtag handling, URL removal, and noise filtering techniaues | We move beyond static tweet analysis and propose a real-time, event-driven system that detects and responds to safety incidents using multi-
---

# Results & Analysis

## Key Findings

* NLP and ML models achieved high sentiment classification accuracy across large-scale Twitter data.
* Negative and unsafe-experience discussions dominated women’s safety-related conversations.
* SVM demonstrated the highest overall performance in sentiment classification.
* Public reaction intensity and harassment mentions showed major spikes during key incidents.
* Trend analysis confirmed the effectiveness of social media data for real-time safety insight extraction.

---
<img width="700" height="500" alt="performance-eval" src="https://github.com/user-attachments/assets/674b6ab6-0b92-4a83-839c-23bfa9025122" />

---
<img width="597" height="346" alt="model-acc" src="https://github.com/user-attachments/assets/cba627d4-a174-4b02-b02f-3b832050141b" />

# Sentiment Distribution

| Sentiment Category  | Percentage |
| ------------------- | ---------- |
| Awareness & Support | 40%        |
| Unsafe Experiences  | 35%        |
| General Discussions | 25%        |

---

# Trend Analysis Dataset

| Month | Fear & Unsafe Discussions | Harassment Mentions | Public Reaction Intensity | Toxic Language Frequency |
| ----- | ------------------------- | ------------------- | ------------------------- | ------------------------ |
| Jan   | 20                        | 18                  | 12                        | 16                       |
| Feb   | 26                        | 25                  | 19                        | 21                       |
| Mar   | 34                        | 31                  | 28                        | 30                       |
| Apr   | 50                        | 47                  | 55                        | 44                       |
| May   | 39                        | 42                  | 46                        | 37                       |
| Jun   | 63                        | 58                  | 60                        | 56                       |
| Jul   | 49                        | 51                  | 52                        | 41                       |

<img width="811" height="412" alt="Trend-analysis" src="https://github.com/user-attachments/assets/fe2c0e69-a751-4ec9-bb80-e10052740be3" />


---

# Technologies Used

| Technology       | Purpose                  |
| ---------------- | ------------------------ |
| Python           | Core Development         |
| NLP              | Text Processing          |
| Machine Learning | Sentiment Classification |
| TF-IDF           | Feature Extraction       |
| Pandas & NumPy   | Data Handling            |
| Matplotlib       | Data Visualization       |
| Scikit-learn     | ML Model Training        |

---

# Real-World Applications

* Women’s safety monitoring systems
* Public sentiment analysis
* Smart city awareness systems
* Social issue trend monitoring
* AI-driven social analytics
* Real-time crisis detection

---

# Future Scope

* Integration of deep learning and transformer-based NLP models
* Real-time social media monitoring dashboards
* Multilingual sentiment analysis
* Geolocation-based safety trend analysis
* AI-powered public safety alert systems

---

# Project Visuals

## Research Poster

> Add your conference poster image here.

```text
visuals/poster.png
```

---

## Results Dashboard



---

# Research Contribution

This research demonstrates how AI and NLP can be leveraged for identifying large-scale public concerns related to women’s safety using social media analytics.

The project combines:

* Social impact
* Artificial Intelligence
* Natural Language Processing
* Real-time sentiment understanding
* Trend-based analytical visualization

---

# Author

## Reda Kaleem

Computer Science Graduate • AI & Systems Developer • Research Enthusiast

---

<div align="center">

### ⭐ If you found this project interesting, consider starring the repository.

</div>
