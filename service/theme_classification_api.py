from transformers import pipeline
from fastapi import FastAPI
from typing import List
import os
import dotenv
import uvicorn
from pydantic import BaseModel, Field
import torch

dotenv.load_dotenv()

# torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize FastAPI app
app = FastAPI()

# Initialize the classifier as a global variable
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
# Create a POST endpoint for classifying text
# Create ClassificationRequest model
class ClassificationRequest(BaseModel):
    text: str = Field(default="Through the meetings, it became clear that the Palestinians tend not to care about the US election results, considering that there is no difference in the policies of the Democratic and Republican parties regarding the Palestinian issue", description="The text to classify")
    candidate_labels: List[str] = Field(default=["Business", "War", "Religion", "Entertainment", "Sport", "Culture", "Travels", "Science", "Education", "Politics"], description="The candidate labels to classify the text into")

@app.post("/classify")
def classify(request: ClassificationRequest):
    result = classifier(request.text, request.candidate_labels)
    return {"labels": result['labels'], "scores": result['scores']}
  
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7555)