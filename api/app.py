from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.train_model import load_model, predict_text  # Model import

# Initialize FastAPI app
app = FastAPI()

# Request schema for text input
class TextInput(BaseModel):
    text: str

# Load model once when API starts — disabled for testing purposes
# Uncomment the line below to enable model loading
# model = load_model()
model = None

@app.on_event("startup")
def startup_event():
    """Optional: Load the model on API startup if needed"""
    global model
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            print(f"Model loading failed: {str(e)}")

@app.post("/predict")
def predict(input: TextInput):
    """
    Endpoint to get predictions based on text input.
    
    Args:
        input (TextInput): JSON payload containing the text string.

    Returns:
        dict: A JSON object with the prediction or an error message.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please enable model loading.")

    try:
        prediction = predict_text(model, input.text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")