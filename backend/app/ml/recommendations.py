def weekly_plan(balance, days_until_income, daily_spend, budgets, spent_by_category, upcoming_mandatory=0):
    days = max(1, days_until_income or 7)
    protected = max(0, upcoming_mandatory)
    runway_budget = max(0, balance - protected) * .72
    weekly = min(runway_budget, daily_spend * min(days, 7)) if daily_spend else runway_budget
    daily = weekly / min(days, 7)
    suggestions = []
    for category, limit in budgets.items():
        spent = spent_by_category.get(category, 0)
        remaining = max(0, limit - spent)
        if spent > limit * .75: suggestions.append({"category": category, "weekly_target": round(min(remaining, weekly * .15), 2), "reason": f"You have used {spent/limit:.0%} of this month's budget; keep this category light."})
        elif category in spent_by_category: suggestions.append({"category": category, "weekly_target": round(min(remaining / max(1, days/7), weekly * .25), 2), "reason": f"₹{remaining:,.0f} remains in your monthly {category.lower()} budget."})
    if not suggestions: suggestions = [{"category": "Flexible spending", "weekly_target": round(weekly, 2), "reason": "No category budgets yet; this cap keeps a buffer for the rest of your runway."}]
    return {"weekly_budget": round(weekly, 2), "daily_limit": round(daily, 2), "days_until_income": days_until_income, "protected_upcoming_expenses": round(protected, 2), "suggestions": suggestions, "explanation": "The plan reserves known mandatory costs and keeps a 28% cushion. Expected income is treated as uncertain.", "disclaimer": "A planning estimate based on your entered data, not guaranteed financial advice."}
