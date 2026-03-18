# Telegram Fake Job Detection & Drift Monitoring Bot

A Telegram bot that automatically detects fake job postings in channel messages using **machine learning** and **rule-based patterns**. It also monitors **model drift** over time to ensure continued accuracy.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Drift Monitoring](#drift-monitoring)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
This project uses a pre-trained ML model and rule-based heuristics to detect fraudulent job postings on Telegram channels. The bot deletes suspicious messages in real-time and logs predictions for **drift monitoring**. Data drift and concept drift are continuously tracked using PSI (Population Stability Index) to identify changes in job posting patterns.

---

## Features
- Detect fake job postings using ML and NLP  
- Rule-based scam detection for fees, deposits, and sensitive document requests  
- Real-time message deletion on Telegram channels  
- Logs predictions for monitoring and retraining  
- PSI-based drift monitoring to detect changes in model performance  

---

## Tech Stack
- **Programming Language:** Python 3  
- **Libraries:** `joblib`, `pandas`, `numpy`, `re`, `python-telegram-bot`  
- **ML Model:** Pre-trained pipeline (`final_fraud_detection_pipeline.pkl`)  
- **Monitoring:** PSI calculation for drift detection  

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-job-telegram-bot.git
   cd fake-job-telegram-bot
2. Create and activate a virtual environment:


# Linux / Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Add your Telegram Bot token in the script:
   BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

5.Run the bot:
    python bot.py

## Usage

- The bot automatically listens to messages in Telegram channels it has access to.
- It predicts whether a job posting is **fake** using ML and rule-based patterns.
- Messages flagged as suspicious are **automatically deleted**.
- All predictions, confidence scores, and message metadata are logged in `predictions_log.csv`.

---

## Drift Monitoring

- Tracks **confidence scores** of recent predictions.
- Calculates **Population Stability Index (PSI)** to detect distribution changes between training and incoming data.
- Alerts when drift occurs:
  - ⚠ **Moderate Drift**: PSI > 0.1
  - 🚨 **Significant Drift**: PSI > 0.25
- Helps retrain the model and adapt to **evolving fraud trends**.

   
