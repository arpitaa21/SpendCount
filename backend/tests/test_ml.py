from app.ml.forecasting import forecast
from app.ml.categorisation import predict_category
from app.ml.recommendations import weekly_plan

def test_forecast_handles_insufficient_history():
    result=forecast([100,200,0],10000,10)
    assert result["predicted_end_balance"] <= 10000
    assert result["confidence"] < .8

def test_categoriser_returns_confidence():
    result=predict_category("coffee cafe lunch")
    assert "confidence" in result and result["confidence"] > 0

def test_plan_never_exceeds_available_balance():
    result=weekly_plan(1000,5,300,{}, {})
    assert result["weekly_budget"] <= 720
