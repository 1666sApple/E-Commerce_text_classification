from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.predict import predict_sentiment

router = APIRouter()

class ReviewInput(BaseModel):
    text: str

@router.post("/predict")
async def predict(review: ReviewInput):
    try:
        sentiment = predict_sentiment(review.text)
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))