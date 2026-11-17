from datetime import date, timedelta
import numpy as np

def forecast(daily_expenses, current_balance, remaining_days, next_income_days=None, expected_income=0):
    """Compare recent moving-average and linear-trend daily burn; choose the lower historical error proxy."""
    vals = np.asarray(daily_expenses, dtype=float)
    vals = np.maximum(vals, 0)
    if len(vals) < 7:
        avg = float(vals.mean()) if len(vals) else 0.0
        recent = avg; uncertainty = .45; model = "moving-average-fallback"
    else:
        avg = float(vals.mean()); recent = float(vals[-14:].mean())
        # Select the approach using a chronological holdout: fit on earlier days,
        # score on the final week, then refit the selected trend on all history.
        cut = len(vals) - 7
        fit_x = np.arange(cut); holdout_x = np.arange(cut, len(vals))
        slope, intercept = np.polyfit(fit_x, vals[:cut], 1)
        trend_holdout = np.maximum(intercept + slope * holdout_x, 0)
        actual = vals[cut:]
        trend_mae = float(np.mean(np.abs(actual - trend_holdout)))
        base_mae = float(np.mean(np.abs(actual - np.repeat(float(vals[max(0,cut-14):cut].mean()), 7))))
        if trend_mae < base_mae * .9:
            slope, intercept = np.polyfit(np.arange(len(vals)), vals, 1)
            recent = float(np.mean(np.maximum(intercept + slope * np.arange(len(vals), len(vals) + 7), 0))); model = "linear-trend-v1"
        else: model = "moving-average-v1"
        uncertainty = min(.6, max(.12, (trend_mae if model.startswith("linear") else base_mae) / max(recent, 1)))
    days = max(0, int(remaining_days)); projected_expenses = recent * days
    no_income = current_balance - projected_expenses
    with_income = no_income + max(0, expected_income)
    next_days = min(days, max(0, int(next_income_days or days)))
    next_balance = current_balance - recent * next_days
    # Safe discretionary amount preserves 15% buffer and known expected income is excluded by default.
    daily_safe = max(0.0, (current_balance - recent * days * .55) / max(days, 1))
    return {"predicted_end_balance": round(no_income, 2), "predicted_with_expected_income": round(with_income, 2), "expected_expenses": round(projected_expenses, 2), "daily_spend_rate": round(recent, 2), "daily_safe_limit": round(daily_safe, 2), "balance_on_next_income_date": round(next_balance, 2), "confidence": round(1 - uncertainty, 2), "uncertainty": round(uncertainty, 2), "model_version": model, "comparison": {"moving_average_daily": round(float(vals[-14:].mean()) if len(vals) else 0, 2), "linear_trend_daily": round(float(recent), 2) if model.startswith("linear") else None}, "days_remaining": days}
