"""Student expense categorisation: transparent keyword rules plus TF-IDF/LogReg."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

TRAINING = {
    "Food": ["canteen lunch", "coffee cafe", "grocery market", "pizza dinner", "mess food", "restaurant", "snacks", "swiggy meal", "chai", "groceries", "campus cafe meal", "dining hall food", "breakfast sandwich", "lunch thali", "dinner noodles", "food delivery order", "fruit and vegetables", "bakery pastry", "tea and biscuits", "meal at cafeteria", "campus food court", "milk and bread"],
    "Transport": ["metro recharge", "bus ticket", "auto ride", "cab uber", "petrol", "train fare", "transport pass", "city bus fare", "metro card top up", "rickshaw fare", "taxi to campus", "ola ride", "fuel station", "commute ticket", "local train pass", "bike taxi", "airport shuttle", "bus pass renewal", "cab to hostel", "parking fee"],
    "Rent and Hostel": ["hostel fee", "room rent", "pg accommodation", "electricity bill", "hostel mess fee", "monthly hostel payment", "rent for student room", "paying guest rent", "dormitory charges", "water bill hostel", "shared flat rent", "hostel deposit", "room maintenance fee", "wifi hostel bill", "accommodation payment", "monthly rent transfer", "campus housing", "electricity hostel", "pg monthly fee", "dorm rent"],
    "Shopping": ["clothes store", "online shopping", "shoes", "amazon order", "electronics", "shopping mall", "new jeans purchase", "backpack shopping", "phone case order", "clothing purchase", "sports shoes", "online store order", "campus supplies shopping", "headphones purchase", "t shirt store", "stationery pouch purchase", "shopping marketplace", "winter jacket", "new clothes", "fashion store"],
    "Subscriptions": ["spotify premium", "netflix subscription", "cloud storage", "monthly subscription", "youtube premium", "music app plan", "video streaming plan", "monthly app renewal", "prime membership", "subscription renewal fee", "online storage plan", "streaming service payment", "music membership", "software monthly plan", "student subscription", "digital magazine renewal", "premium app charge", "monthly platform fee", "cloud backup renewal", "entertainment subscription"],
    "Academics": ["textbook", "course materials", "exam fee", "stationery", "college lab", "online course", "printing notes", "university exam registration", "study guide purchase", "lab coat for class", "notebook and pens", "photocopy lecture notes", "academic journal access", "course registration fee", "college project supplies", "reference book", "semester tuition payment", "assignment printing", "calculator for class", "educational workshop fee"],
    "Entertainment": ["movie ticket", "concert", "gaming", "weekend outing", "bowling", "event ticket", "cinema tickets", "live music show", "arcade games", "campus festival ticket", "theatre show", "escape room outing", "comedy night tickets", "museum visit", "gaming cafe", "weekend movie", "music event entry", "theme park pass", "board game cafe", "social event ticket"],
    "Healthcare": ["pharmacy medicine", "doctor consultation", "clinic", "medical test", "health checkup", "prescription medicine", "dentist appointment", "cold medicine purchase", "doctor visit fee", "hospital consultation", "eye checkup", "first aid supplies", "health clinic bill", "medical prescription", "blood test charge", "pharmacy vitamins", "lab diagnostic fee", "urgent care visit", "vaccination fee", "medical bandage"],
    "Personal Care": ["haircut salon", "toiletries", "skincare", "laundry", "personal grooming", "barber haircut", "shampoo and soap", "face wash purchase", "laundry service", "toothpaste toiletries", "salon appointment", "skin lotion", "deodorant purchase", "dry cleaning clothes", "personal hygiene items", "hair oil and comb", "nail care salon", "body wash", "washing clothes service", "grooming kit"],
    "Other": ["miscellaneous payment", "bank charge", "other expense", "uncategorized", "small service fee", "administrative charge", "misc payment", "one time fee", "unclassified transaction", "document courier charge", "unexpected expense", "service processing fee", "general payment", "miscellaneous charge", "account fee", "unknown purchase", "cash withdrawal fee", "other campus cost", "payment fee", "general expense"],
}

MODEL_VERSION = "spendcount-tfidf-logreg-sample-v2"
MODEL_PATH = Path(__file__).parent / "models" / "categoriser.joblib"
SAMPLE_PATH = Path(__file__).resolve().parents[3] / "dataset" / "sample_data" / "student_transactions_6_months.csv"

# The supplied fixture uses a few labels that differ from the app taxonomy.
SAMPLE_CATEGORY_MAP = {
    "Hostel": "Rent and Hostel",
    "Utilities": "Other",
}

# High-precision, explainable merchant/phrase cues take precedence over the
# statistical model. These are a routing aid, not learned confidence scores.
CATEGORY_RULES = (
    ("Rent and Hostel", ("hostel", "accommodation", "dorm", "room rent", "pg rent")),
    ("Entertainment", ("bowling", "college fest", "fest entry", "movie", "cinema", "gaming", "concert", "outing", "event ticket")),
    ("Transport", ("bus", "cab", "auto", "train", "metro", "taxi", "fare", "commute")),
    ("Subscriptions", ("subscription", "spotify", "cloud storage", "premium", "membership")),
    ("Academics", ("course", "lab", "notebook", "photocopy", "printout", "semester notes", "reference book", "textbook", "stationery", "keyboard replacement")),
    ("Food", ("canteen", "cafe", "meal", "lunch", "breakfast", "snack", "fruit", "grocery", "milk", "swiggy", "zomato", "tea")),
    ("Shopping", ("gift", "clothing", "clothes", "shoes", "screen repair", "shopping", "phone case", "purchase")),
    ("Other", ("mobile recharge", "utility", "bank charge", "service fee")),
)


def normalize_description(description: str) -> str:
    return " ".join(str(description).casefold().split())


def load_sample_examples() -> list[tuple[str, str]]:
    """Load unique, valid expense descriptions only; ignore income/test fixtures."""
    if not SAMPLE_PATH.exists():
        return []
    examples: dict[str, str] = {}
    with SAMPLE_PATH.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("transaction_type", "")).strip().casefold() != "expense":
                continue
            category = str(row.get("category", "")).strip()
            category = SAMPLE_CATEGORY_MAP.get(category, category)
            description = str(row.get("description", "")).strip()[:250]
            if not description or category not in TRAINING:
                continue
            examples.setdefault(normalize_description(description), category)
    return sorted(examples.items())


def _training_corpus(corrections=(), sample_examples=None):
    texts: list[str] = []
    labels: list[str] = []
    for category, examples in TRAINING.items():
        texts.extend(examples)
        labels.extend([category] * len(examples))
    for description, category in (load_sample_examples() if sample_examples is None else sample_examples):
        texts.append(description)
        labels.append(category)
    for description, category in corrections:
        normalized = str(description).strip()[:250]
        label = str(category).strip()
        if normalized and label in TRAINING:
            texts.append(normalized)
            labels.append(label)
    return texts, labels


def build_model(corrections=(), sample_examples=None):
    texts, labels = _training_corpus(corrections, sample_examples)
    model = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0, random_state=17)),
    ])
    model.fit(texts, labels)
    return model


def train_model():
    """Train final model and save a versioned joblib bundle for deployment."""
    model = build_model()
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": MODEL_VERSION, "model": model}, MODEL_PATH)
    except OSError:
        pass
    return model


@lru_cache(maxsize=1)
def _model():
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict) and bundle.get("version") == MODEL_VERSION:
                return bundle["model"]
        except Exception:
            pass
    return train_model()


@lru_cache(maxsize=32)
def _model_with_corrections(corrections: tuple[tuple[str, str], ...]):
    return build_model(corrections=corrections)


def predict_income_category(description: str):
    text = normalize_description(description)
    rules = [
        ("Scholarship", ("scholarship", "grant", "bursary", "stipend award")),
        ("Internship", ("internship", "intern stipend", "intern pay")),
        ("Part-time Job", ("part time", "part-time", "salary", "wage", "payroll", "shift pay", "job pay")),
        ("Freelancing", ("freelance", "client payment", "project payment", "gig income")),
        ("Family Allowance", ("allowance", "from parents", "family transfer", "parent transfer", "pocket money")),
    ]
    for category, terms in rules:
        if any(term in text for term in terms):
            return {"category": category, "predicted_category": category, "confidence": 0.9, "method": "keyword_rule", "explanation": f"The description contains an income cue associated with {category}."}
    return {"category": "Needs Review", "predicted_category": "Other Income", "confidence": 0.24, "method": "fallback", "explanation": "No supported income-source cue matched; review the category."}


def _rule_prediction(description: str):
    text = normalize_description(description)
    for category, cues in CATEGORY_RULES:
        cue = next((item for item in cues if item in text), None)
        if cue:
            return {"category": category, "predicted_category": category, "confidence": 0.92, "method": "keyword_rule", "explanation": f"Matched the phrase cue '{cue}' for {category}."}
    return None


def predict_category(description: str, corrections=()):
    normalized = normalize_description(description)
    correction_data = tuple((str(text), str(category)) for text, category in corrections)
    # The student's exact correction takes precedence over global heuristics.
    for corrected_description, category in reversed(correction_data):
        if normalize_description(corrected_description) == normalized and category in TRAINING:
            return {"category": category, "predicted_category": category, "confidence": 0.99, "method": "user_correction", "explanation": "Matched a category correction you previously made."}
    rule = _rule_prediction(description)
    if rule:
        return rule
    model = _model_with_corrections(correction_data) if correction_data else _model()
    probs = model.predict_proba([description])[0]
    order = probs.argsort()
    idx = int(order[-1])
    confidence = float(probs[idx])
    margin = confidence - float(probs[order[-2]]) if len(order) > 1 else confidence
    category = str(model.classes_[idx])
    accepted = confidence >= 0.38 and margin >= 0.06
    return {
        "category": category if accepted else "Needs Review",
        "predicted_category": category,
        "confidence": round(confidence, 3),
        "method": "tfidf_logistic_regression",
        "explanation": f"The labelled text model ranked {category} highest (score {confidence:.0%}); {'' if accepted else 'the score or lead over the next category is too small, so review is needed.'}",
    }
