import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
import numpy as np
import torch


# ============================================================
# 1) Load Local SQuAD JSON
# ============================================================
def load_local_squad(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    contexts, questions, answers = [], [], []
    articles = raw["data"]

    for article in articles:
        context = article["context"]
        question = article["question"]

        if len(article["answer_text"]) == 0:
            ans_text = ""
            ans_start = -1
        else:
            ans_text = article["answer_text"]
            ans_start = article["answer_start"]

        contexts.append(context)
        questions.append(question)
        answers.append({"text": ans_text, "answer_start": ans_start})

    return Dataset.from_dict({
        "context": contexts,
        "question": questions,
        "answers": answers,
    })


print("Loading local SQuAD...")
train_ds = load_local_squad("dev_100.json")
val_ds   = load_local_squad("train_100.json")

raw_datasets = DatasetDict({
    "train": train_ds,
    "validation": val_ds
})


# ============================================================
# 2) Tokenization + Create Start/End Token Labels
# ============================================================
model_name = "google/electra-small-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def prepare_train_features(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_map = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_map):
        sample_idx = sample_map[i]
        answer = examples["answers"][sample_idx]

        if answer["answer_start"] == -1:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        seq_ids = tokenized.sequence_ids(i)

        context_start = seq_ids.index(1)
        context_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)

        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= start_char:
                token_start += 1
            token_start -= 1

            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            token_end += 1

            start_positions.append(token_start)
            end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


print("Tokenizing...")
tokenized_train = raw_datasets["train"].map(
    prepare_train_features,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

tokenized_val = raw_datasets["validation"].map(
    prepare_train_features,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names
)


# ============================================================
# 3) Accuracy Metric for QA (exact span match)
# ============================================================
def compute_metrics(p):
    start_logits, end_logits = p.predictions
    start_labels = p.label_ids["start_positions"]
    end_labels = p.label_ids["end_positions"]

    pred_start = np.argmax(start_logits, axis=1)
    pred_end   = np.argmax(end_logits, axis=1)

    accuracy = np.mean((pred_start == start_labels) & (pred_end == end_labels))
    return {"accuracy": accuracy}


# ============================================================
# 4) Load ELECTRA model
# ============================================================
print("Loading ELECTRA model...")
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


# ============================================================
# 5) Training Arguments
# ============================================================
training_args = TrainingArguments(
    output_dir="electra_squad_local",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,        # change if needed
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
)


# ============================================================
# 6) Trainer
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,   # <-- accuracy added
)


# ============================================================
# 7) Train
# ============================================================
print("Training...")
trainer.train()

trainer.save_model("electra_squad_local")
print("Model saved to electra_squad_local/")