from collections import Counter, defaultdict
from datetime import date, timedelta
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
    # Compare the latest seven days with the preceding four weeks to catch
    # category-level changes that a transaction amount model cannot see.
    today=date.today(); current_start=today-timedelta(days=6); baseline_start=current_start-timedelta(days=28)
    current=defaultdict(float); baseline=defaultdict(float); current_count=Counter(); baseline_count=Counter()
    for tx in expenses:
        if current_start<=tx.date<=today:
            current[tx.category]+=tx.amount; current_count[tx.category]+=1
        elif baseline_start<=tx.date<current_start:
            baseline[tx.category]+=tx.amount; baseline_count[tx.category]+=1
    for category,amount in current.items():
        typical=baseline[category]/4
        if typical>=100 and amount>typical*1.3 and amount-typical>=100:
            result.append({"transaction_id":f"weekly-spike-{category}","date":today.isoformat(),"description":f"{category} spending this week","category":category,"amount":round(amount,2),"typical_amount":round(typical,2),"reason":f"Your {category.lower()} spending is {((amount/typical)-1)*100:.0f}% above its average week over the previous four weeks. This is a pattern to review, not a fraud finding."})
        usual_count=baseline_count[category]/4
        if usual_count>=1 and current_count[category]>=3 and current_count[category]>usual_count*2.5 and not (typical>=100 and amount>typical*1.3 and amount-typical>=100):
            result.append({"transaction_id":f"weekly-frequency-{category}","date":today.isoformat(),"description":f"More {category.lower()} transactions this week","category":category,"amount":round(amount,2),"typical_amount":round(typical,2),"reason":f"You recorded {current_count[category]} {category.lower()} expenses this week, compared with about {usual_count:.1f} in a typical week. Check whether this is a temporary change."})
    result.sort(key=lambda row: row["amount"],reverse=True)
    return result
