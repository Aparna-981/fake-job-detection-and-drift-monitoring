import joblib
import pandas as pd
import numpy as np
import re
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ==============================
# SETTINGS
# ==============================

BOT_TOKEN = ""
CONFIDENCE_THRESHOLD = 0.1

# ==============================
# LOAD MODEL (Full Pipeline)
# ==============================

model = joblib.load("final_fraud_detection_pipeline.pkl")
training_confidence = np.load("training_confidence.npy")

# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_job(text):
    prediction = model.predict([text])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = probabilities[prediction]

    elif hasattr(model, "decision_function"):
        score = model.decision_function([text])[0]

        # If array, take first value
        if isinstance(score, (list, np.ndarray)):
            score = score[0]

        # Convert to probability using sigmoid
        confidence = 1 / (1 + np.exp(-float(score)))

    else:
        confidence = 0.0

    return int(prediction), float(confidence)


# ==============================
# RULE-BASED SCAM DETECTION
# ==============================

def has_fee_scam_pattern(text):
    text = text.lower()
    suspicious_patterns = [
        "security deposit",
        "registration fee",
        "refundable deposit",
        "processing fee",
        "submit id",
        "laptop dispatch",
        "aadhaar",
        "passport copy"
    ]
    return any(pattern in text for pattern in suspicious_patterns)


# ==============================
# LOGGING (for drift monitoring)
# ==============================

def clean_text(text):
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def log_prediction(title, description, prediction, confidence):
    clean_title = clean_text(title)
    clean_description = clean_text(description)

    data = {
        "timestamp": [datetime.now()],
        "title": [clean_title],
        "description": [clean_description],
        "prediction": [prediction],
        "confidence": [round(float(confidence), 4)]
    }

    df = pd.DataFrame(data)

    df.to_csv(
        "predictions_log.csv",
        mode="a",
        header=not pd.io.common.file_exists("predictions_log.csv"),
        index=False,
        encoding="utf-8"
    )


# ==============================
# HANDLE NEW CHANNEL MESSAGE
# ==============================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    message = update.channel_post

    # Safety check
    if not message or not message.text:
        return

    text = message.text

    # Extract title (first line)
    lines = text.split("\n")
    title = lines[0]
    description = text

    # Run prediction ONCE
    prediction, confidence = predict_job(text)
    rule_flag = has_fee_scam_pattern(text)

    print("Prediction:", prediction)
    print("Confidence:", confidence)
    print("Rule flag:", rule_flag)
    print("----------------------")

    # Log everything
    log_prediction(title, description, prediction, confidence)

    # ================= DECISION LOGIC =================
    check_drift()

    delete_message = False

    # Rule-based detection (highest priority)
    if rule_flag:
        print("Rule-based scam detected → delete")
        delete_message = True

    # ML fraud detection
    elif prediction == 1 and confidence >= CONFIDENCE_THRESHOLD:
        print("ML detected fraud with high confidence → delete")
        delete_message = True

    # Legit & confident
    else:
        print("Legit or low confidence → keep")
        return

    # Low confidence safety mode
    
    # ================= DELETE BLOCK =================

    if delete_message:
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=message.message_id
            )
        except Exception as e:
            print("Delete failed:", e)

def calculate_psi(expected, actual, buckets=10):

    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(expected, np.arange(0, 101, 100 / buckets))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    psi = 0

    for i in range(len(breakpoints) - 1):
        e_count = ((expected >= breakpoints[i]) & (expected < breakpoints[i+1])).sum()
        a_count = ((actual >= breakpoints[i]) & (actual < breakpoints[i+1])).sum()

        e_perc = e_count / len(expected)
        a_perc = a_count / len(actual)

        if e_perc > 0 and a_perc > 0:
            psi += (a_perc - e_perc) * np.log(a_perc / e_perc)

    return psi

def check_drift():

    if training_confidence is None:
        return

    if not pd.io.common.file_exists("predictions_log.csv"):
        return

    log_df = pd.read_csv("predictions_log.csv")

    # Only check when enough data collected
    if len(log_df) < 10:
        return

    recent_conf = log_df.tail(10)["confidence"].values

    psi_value = calculate_psi(training_confidence, recent_conf)

    print("Current PSI:", round(psi_value, 4))

    if psi_value > 0.25:
        print("🚨 SIGNIFICANT DRIFT DETECTED")
    elif psi_value > 0.1:
        print("⚠ Moderate Drift")
    else:
        print("✅ No Drift")


# ==============================
# RUN BOT
# ==============================

app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(
    MessageHandler(filters.ChatType.CHANNEL & filters.TEXT, handle_message)
)

print("Bot is running...")

app.run_polling()