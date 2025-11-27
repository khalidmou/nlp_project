import json

with open('train-v1.1.json', 'r', encoding='utf-8') as f:
     squad_data = json.load(f)

 # SQuAD structure: {"data": [{"title":..., "paragraphs":[{"context":..., "qas":[...]}, ...]}, ...]}
 # We'll flatten to get individual question-answer-context triples

# records = []
# for article in squad_data['data']:
#      for paragraph in article['paragraphs']:
#          context = paragraph['context']
#          for qa in paragraph['qas']:
#              question = qa['question']
#              answers = qa['answers']  # List of answers (usually one for v1.1)
#              for answer in answers:
#                  record = {
#                      'context': context,
#                      'question': question,
#                      'answer_text': answer['text'],
#                      'answer_start': answer['answer_start']
#                  }
#                  records.append(record)
#              if len(records) >= 2000:  # Stop after 100 records
#                  break
#          if len(records) >= 2000:
#              break
#      if len(records) >= 2000:
#          break

#  # Save first 100 records to a new JSON file
# with open('train_2000.json', 'w', encoding='utf-8') as f:
#     json.dump(records[:2000], f, ensure_ascii=False, indent=2)

# print("Saved first 2000 records to 'squad_first_2000.json'")





from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Path to your fine-tuned model directory
model_path = "./fine_tuned_electra_qa"  # change this to your folder

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

context = "Super Bowl 50 was an American football game won by the Denver Broncos."
question = "Which team won Super Bowl 50?"

inputs = tokenizer(
    question,
    context,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

# outputs contains start_logits and end_logits
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Get the most likely start and end token positions
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits)

# Convert tokens to answer text
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx+1])
)

print("Question:", question)
print("Answer:", answer)