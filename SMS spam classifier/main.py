from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load your model and vectorizer at startup
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("spam_vectorizer.joblib")

class SMSRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_spam(request: SMSRequest):
    # 1. Clean/preprocess text here if needed
    cleaned_text = [request.text] 
    
    # 2. Transform text using the saved training vectorizer
    vectorized_text = vectorizer.transform(cleaned_text)
    
    # 3. Predict
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0][1] # Probability of spam
    
    return {
        "label": "spam" if prediction == 1 else "ham",
        "spam_probability": float(probability)
    }