from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="E-commerce Review Classifier")

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the E-commerce Review Classifier API"}