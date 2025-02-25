from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow only frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Define request body model
class QuestionRequest(BaseModel):
    question: str

# Load model and tokenizer from checkpoint
MODEL_PATH = "checkpoint-11000"  
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)

@app.post("/generate/")
async def generate_answer(request: QuestionRequest):
    input_text = request.question

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_length=100)

    # Decode output
    answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return {"question": input_text, "answer": answer}

# Run server: uvicorn main:app --reload
