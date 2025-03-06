from fastapi import FastAPI
from pydantic import BaseModel
from src.train_model import load_model, predict_text  # Beispiel: Modellimport

app = FastAPI()

# Request schema
class TextInput(BaseModel):
    text: str

# Load model once when API starts
model = load_model()

@app.post("/predict")
def predict(input: TextInput):
    prediction = predict_text(model, input.text)
    return {"prediction": prediction}