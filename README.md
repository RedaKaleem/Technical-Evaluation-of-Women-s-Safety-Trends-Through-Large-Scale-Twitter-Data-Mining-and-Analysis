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

# Orange Workflow

<img width="1375" height="604" alt="orange-workflow" src="https://github.com/user-attachments/assets/48c33e8c-d5f6-4541-abd6-efb94c5776e1" />

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

| Technology              | Purpose                    |
| ----------------        | ------------------------   |
| Python                  | Core Development           |
| NLP                     | Text Processing            |
| Machine Learning        | Sentiment Classification   |
| TF-IDF                  | Feature Extraction         |
| Pandas & NumPy          | Data Handling              |
| Matplotlib              | Data Visualization         |
| Scikit-learn            | ML Model Training          |
| Orange Data Mining Tool | Model training and analysis|

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

## References
1. Mustapha, W.N.A.W., et al.: Detection of harassment toward women in Twitter during pandemic based on machine learning. Int. J. Adv. Comput. Sci. Appl. 15(3) (2024)
1. Albladi, A., Islam, M.: Sentiment analysis of Twitter data using NLP models. IEEE Access (2025)
1. Patil, H., et al.: Quantitative sentiment analysis of women’s safety. J. Inf. Syst. (2025)
1. Sablok, B., et al.: NLP models for sentiment analysis. IJRASET (2025)
1. Rodríguez-Sánchez, F., et al.: Classification of sexism in social networks. IEEE Access (2020)
1. Qi, Y., Shabrina, Z.: Sentiment analysis using Twitter data. Soc. Netw. Anal. Min. (2023)
1. Wang, Y., et al.: Sentiment analysis of Twitter data. Appl. Sci. (2022)
1. Basha, A., et al.: WomenSafe AI. AI Cyber Comput. Manag. (2022)
1. Lokesh, U., et al.: Women safety analysis in Indian cities. (2019)
1. Lagrari, F.E., ElKettani, Y.: BERT for sentiment analysis. Springer (2023)
1. Qaisar, S.M.: Sentiment analysis using LSTM. (2020)
1. Jahin, M.A., et al.: Hybrid transformer model. arXiv (2024)
1. Kashid, S., et al.: Bi-RNN text classification. Springer (2022)
1. F.-E. Lagrari and Y. ElKettani, ‘‘A comparative study of a new customizedBERT for sentiment analysis,’’ in Sentiment Analysis and Deep Learning.Berlin, Germany: Springer, 2023, pp. 315–322.
1. S. M. Qaisar, ‘‘Sentiment analysis of IMDb movie reviews using longshort-term memory,’’ in Proc. 2nd Int. Conf. Comput. Inf. Sci. (ICCIS),Oct. 2020, pp. 1–4.
1. M. A. Jahin, M. S. H. Shovon, M. F. Mridha, M. R. Islam, andY. Watanobe, ‘‘A hybrid transformer and attention based recurrent neuralnetwork for robust and interpretable sentiment analysis of tweets,’’ 2024,arXiv:2404.00297.
1. S. Kashid, K. Kumar, P. Saini, A. Dhiman, and A. Negi, ‘‘Bi-RNN and bi-LSTM based text classification for Amazon reviews,’’ in Proc. Int. Conf.Deep Learn., Artif. Intell. Robot. Springer, 2022, pp. 62–72.
1. S. Alipour, A. Galeazzi, E. Sangiorgio, M. Avalle, L. Bojic, M. Cinelli,and W. Quattrociocchi, ‘‘Cross-platform social dynamics: An analysis ofChatGPT and COVID-19 vaccine conversations,’’ Sci. Rep., vol. 14, no. 1,p. 2789, Feb. 2024.
1. A. Amini, Y. E. Bayiz, A. Ram, R. Marculescu, and U. Topcu,‘‘News source credibility assessment: A Reddit case study,’’ 2024,arXiv:2402.10938.
1. A. Sittar, D. Mladenić, and M. Grobelnik, ‘‘Profiling the barriers to thespreading of news using news headlines,’’ Frontiers Artif. Intell., vol. 6,Aug. 2023, Art. no. 1225213.
1. M. Qorich and R. El Ouazzani, ‘‘Text sentiment classification ofAmazon reviews using word embeddings and convolutional neu-ral networks,’’ J. Supercomput., vol. 79, no. 10, pp. 11029–11054,Jul. 2023.VOLUME 13, 2025 30465A. Albladi et al.: Sentiment Analysis of Twitter Data Using NLP Models: A Comprehensive Review
1. F. Nadi, H. Naghavipour, T. Mehmood, A. B. Azman,J. A. P. Nagantheran, K. S. K. Ting, N. M. I. B. N. Adnan,R. A. P. Sivarajan, S. A. P. Veerah, and R. F. Rahmat, ‘‘Sentimentanalysis using large language models: A case study of GPT-3.5,’’ in Proc.Int. Conf. Data Sci. Emerg. Technol. Springer, 2024, pp. 161–168.
1. J. Jiang, X. Ren, and E. Ferrara, ‘‘Retweet-BERT: Political leaningdetection using language features and information diffusion on socialnetworks,’’ in Proc. Int. AAAI Conf. Web Social Media, vol. 17, Jun. 2023,pp. 459–469.
1. Jonnala, N.S., et al.:Leveraging Hybrid Model for Accurate Sentiment Analysis of Twitter Data.Scientific Reports, 2025.
1. Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.:BERT: Pre-training of deep bidirectional transformers for language understanding.Proceedings of NAACL-HLT, pp. 4171–4186 (2019)
1. Davidson, T., Warmsley, D., Macy, M., Weber, I.:Automated hate speech detection and the problem of offensive language.Proceedings of the International AAAI Conference on Web and Social Media (ICWSM), pp. 512–515 (2017)



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
