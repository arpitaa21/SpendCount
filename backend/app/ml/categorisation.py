from pathlib import Path
from functools import lru_cache
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
 "Other": ["miscellaneous payment", "bank charge", "other expense", "uncategorized", "small service fee", "administrative charge", "misc payment", "one time fee", "unclassified transaction", "document courier charge", "unexpected expense", "service processing fee", "general payment", "miscellaneous charge", "account fee", "unknown purchase", "cash withdrawal fee", "other campus cost", "payment fee", "general expense"]
}
MODEL_PATH = Path(__file__).parent / "models" / "categoriser.joblib"

def predict_income_category(description: str):
    text=description.lower()
    rules=[("Scholarship",("scholarship","grant","bursary","stipend award")),("Internship",("internship","intern stipend","intern pay")),("Part-time Job",("part time","part-time","salary","wage","payroll","shift pay","job pay")),("Freelancing",("freelance","client payment","project payment","gig income")),("Family Allowance",("allowance","from parents","family transfer","parent transfer","pocket money"))]
    for category,terms in rules:
        if any(term in text for term in terms): return {"category":category,"predicted_category":category,"confidence":0.9}
    return {"category":"Needs Review","predicted_category":"Other Income","confidence":0.24}

def train_model():
    texts, labels = [], []
    for category, examples in TRAINING.items():
        texts.extend(examples); labels.extend([category] * len(examples))
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")), ("classifier", LogisticRegression(max_iter=500, class_weight="balanced"))])
    model.fit(texts, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, MODEL_PATH)
    return model

@lru_cache(maxsize=1)
def _model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train_model()

def predict_category(description: str):
    model = _model()
    probs = model.predict_proba([description])[0]
    idx = probs.argmax(); confidence = float(probs[idx]); category = str(model.classes_[idx])
    return {"category": category if confidence >= .27 else "Needs Review", "predicted_category": category, "confidence": round(confidence, 3)}
