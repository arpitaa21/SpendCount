import numpy as np
from sklearn.ensemble import IsolationForest

def find_anomalies(transactions):
    expenses = [t for t in transactions if t.transaction_type == "expense"]
    if len(expenses) < 8: return []
    # Features are log amount + category frequency context; model is fitted per-user.
    amounts = np.array([[np.log1p(t.amount)] for t in expenses])
    model = IsolationForest(contamination=min(.12, max(.04, 2 / len(expenses))), random_state=17)
    labels = model.fit_predict(amounts)
    result = []
    for tx, label, score in zip(expenses, labels, model.decision_function(amounts)):
        if label == -1:
            peers = [t.amount for t in expenses if t.category == tx.category and t.id != tx.id]
            typical = float(np.median(peers)) if peers else float(np.median([t.amount for t in expenses]))
            result.append({"transaction_id": tx.id, "date": tx.date.isoformat(), "description": tx.description, "category": tx.category, "amount": tx.amount, "typical_amount": round(typical, 2), "reason": f"This is unusually large compared with your past {tx.category.lower()} transactions (median ₹{typical:,.0f}). This is a spending outlier, not a fraud finding."})
    return result
