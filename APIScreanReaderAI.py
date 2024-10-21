from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Initialize FastAPI app
app = FastAPI()

# Load your trained model and tokenizer
model_path = "./trained_jaws_model"  # Update with the path where your model is saved
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

@app.post("/ask_jaws/")
async def ask_jaws(question: str):
    # Encode the user's question
    input_ids = tokenizer.encode(question, return_tensors='pt')
    # Generate a response
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    # Decode the output and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}

# To run the server: uvicorn backend:app --reload
