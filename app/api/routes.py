from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.predict import predict_sentiment

router = APIRouter()

class ReviewInput(BaseModel):
    text: str

@router.post("/predict")
async def predict(review: ReviewInput):
    try:
        sentiment_id = predict_sentiment(review.text)
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Very Positive"}
        sentiment = sentiment_map.get(sentiment_id, "Unknown")
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))