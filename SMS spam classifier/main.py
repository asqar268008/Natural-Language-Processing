from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("spam_vectorizer.joblib")

class SMSRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_spam(request: SMSRequest):
    cleaned_text = [request.text] 
    
    vectorized_text = vectorizer.transform(cleaned_text)
    
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0][1] # Probability of spam
    
    return {
        "label": "harmful" if prediction == "ham" else "spam",
        "spam_probability": float(probability)
    }
