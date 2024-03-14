from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = Flask(__name__)

# Load the tokenizer and model from the Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = AutoModelForQuestionAnswering.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# Assuming the PDF is named 'fixed_document.pdf' and is in the same directory as app.py
pdf_file_path = 'fixed_document.pdf'

@app.route('/answer', methods=['POST'])
def get_answer():
    content = request.json
    question = content['question']
    
    # Implement the logic to extract the context from your PDF
    # For example, using a PDF parsing library like PyMuPDF or PDFMiner
    context = 'Extracted context from your PDF would go here'
    
    # Encode the question and context
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    # Get the scores for the start and end of the answer
    answer_start_scores, answer_end_scores = model(**inputs)

    # Find the tokens with the highest start and end scores
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert the tokens to a string for the answer
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    # Return the answer as JSON
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
