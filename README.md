# 🧠 AI Health Monitoring System

A multi-agent AI system that fuses emotional text analysis with physiological wearable data to assess mental health risk levels. The system combines NLP-based sentiment analysis with wearable health metrics to generate interpretable and supportive mental health insights.

---

## Features

- **Multi-Agent Architecture**
  - **Text Agent**: Emotion detection from text using a fine-tuned DistilBERT model
  - **Wearable Agent**: Stress prediction from physiological metrics using XGBoost
  - **Fusion Agent**: Weighted fusion of text and wearable outputs for holistic risk assessment

- **Interactive Web Interface**
  - Built with Gradio for real-time simulation

- **Risk Categorization**
  - Low, Mild, Moderate, High Risk

- **Explainable Outputs**
  - Human-readable warnings, advice, and comforting messages

- **Result Logging**
  - Longitudinal tracking using JSONL format

---

## System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web Interface                    │
│                      (simulate.py)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Fusion Agent                            │
│          Combines text + wearable analysis                  │
│          Calculates risk score & provides advice            │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐          ┌──────────────────────────┐
│    Text Agent        │          │   Wearable Agent         │
│  DistilBERT Model    │          │   XGBoost Classifier     │
│  Sentiment Analysis  │          │   Stress Prediction      │
└──────────────────────┘          └──────────────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐          ┌──────────────────────────┐
│  Social Media/       │          │  Health Metrics:         │
│  Journal Text        │          │  - Heart Rate            │
│                      │          │  - Sleep Duration        │
│                      │          │  - Steps, Calories       │
└──────────────────────┘          └──────────────────────────┘
```
---
### Clone the Repository

```bash
git clone https://github.com/Swetha-Arul/AI-Mental-Health-Monitoring-System.git
cd AI-Mental-Health-Monitoring-System
```
---
### Create Virtual Environment (Recommended)

```python -m venv venv```

-**Windows**

```venv\Scripts\activate```


-**Linux / macOS**
```
source venv/bin/activate
```
---
-**Install Dependencies**
```
pip install -r requirements.txt
```
---
### Usage
Run the Gradio Interface
```
python simulate.py
```
---
### 📁Project Structure
```
AI-Mental-Health-Monitoring-System/
├── simulate.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── text_agent/
│   ├── main.py
│   └── sentiment-analysis-model/   # excluded from Git (large files)
│
├── wearable_agent/
│   ├── main.py
│   ├── stress_model_xgb.pkl        
│   ├── scaler.pkl                  
│   └── label_encoder.pkl           
│
├── fusion_agent/
│   ├── main.py
│   └── results_log.jsonl
```
---
## Agents Overview

- **Text Agent**
  - Emotion and sentiment detection from user-provided text
  - Built using a fine-tuned **DistilBERT** model from Hugging Face
  - Classifies text into emotional states such as *normal, anxiety, depression, suicidal, fear, anger*
  - Outputs a predicted label along with a confidence score
  - Designed to work on short, informal text (journals, social media, chat messages)

- **Wearable Agent**
  - Stress level prediction based on physiological and activity data
  - Implemented using an **XGBoost** classifier with preprocessed features
  - Uses metrics such as heart rate, sleep duration, sleep efficiency, steps, and calories
  - Outputs a stress category (*Low, Mild, Moderate, High*) with class probabilities
  - Includes rule-based fallback logic when model files are unavailable

- **Fusion Agent**
  - Combines outputs from the Text Agent and Wearable Agent
  - Uses a weighted fusion strategy (default: 60% text, 40% wearable)
  - Produces a single interpretable mental health risk score
  - Maps the score to final risk levels (*Low, Mild, Moderate, High*)
  - Generates contextual explanations, warnings, and personalized advice
---
### ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **not** a medical diagnostic tool.

If you are experiencing emotional distress, please seek help from a qualified professional or local helpline.
