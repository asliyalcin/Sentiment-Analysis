# model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import yaml



with open("utils/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["models"]["sentiment"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def binary_probs_to_score_1_5(p_pos: float) -> float:
    return 1 + 4 * p_pos


def sentiment_class_from_score(score: float) -> str:
    if score < 2.5:
        return "negative"
    elif score < 3.5:
        return "neutral"
    else:
        return "positive"


def predict_sentiment_batch(text_list, batch_size=32):
    """
    df["Cleaned_Feedback"] gibi önceden temizlenmiş bir liste alır.

    """

    all_pos = []
    all_neg = []
    all_score = []
    all_label = []


    cleaned_texts = ["" if t is None else str(t) for t in text_list]

    for i in range(0, len(cleaned_texts), batch_size):
        batch_texts = cleaned_texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)  # (batch, 2)

        pos_batch = probs[:, 1].cpu().tolist()
        neg_batch = probs[:, 0].cpu().tolist()


        score_batch = [1 + 4 * p for p in pos_batch]
        label_batch = [sentiment_class_from_score(s) for s in score_batch]

        all_pos.extend(pos_batch)
        all_neg.extend(neg_batch)
        all_score.extend(score_batch)
        all_label.extend(label_batch)

    return {
        "pos": all_pos,
        "neg": all_neg,
        "score": all_score,
        "label": all_label
    }
def get_sentiment_probs(text: str):
    """
    TEK bir metin için olasılık döndürür.
    {'neg': 0.xx, 'pos': 0.yy} formatında.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).flatten()

    # 0: negatif, 1: pozitif
    return {
        "neg": float(probs[0]),
        "pos": float(probs[1]),
    }


def mismatch_type(row):
    if row["sent_score"] >= 3.5 and row["Score"] <= 2:
        return "Positive_Text_Low_Score"
    elif row["sent_score"] <= 2.5 and row["Score"] >= 4:
        return "Negative_Text_High_Score"
    else:
        return "Consistent"
