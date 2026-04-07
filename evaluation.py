import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForTokenClassification
)
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_from_disk


# LOAD MODEL AND TOKENIZER
model_path = "LorenzoVentrone/SentenceSplitter-it-en"
MAX_LENGTH = 128
print(f"Loading model {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

test_dataset = load_from_disk("testset")

# Function to tokenize words and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        max_length=MAX_LENGTH
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Ignore special tokens (CLS, SEP, PAD)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # Assign label to the first sub-token of the word
            else:
                label_ids.append(-100) # Ignore subsequent sub-tokens of the same word
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing the Test Set...")
# Map the dataset using the function we just created
tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

# The Data Collator handles dynamic padding of the batches
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# CONFIGURE TRAINER AND PREDICT
eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=16,
    do_train=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=eval_args,
    data_collator=data_collator,
)

print("\n--- STARTING PREDICTION ON TEST SET ---")
# Use the tokenized dataset, not the raw one!
raw_predictions = trainer.predict(tokenized_test_dataset)

# EXTRACT AND CLEAN RESULTS

logits = raw_predictions.predictions
labels = raw_predictions.label_ids

predictions = np.argmax(logits, axis=2)

# Clean up ignored tokens (-100)
true_predictions = [p for (p, l) in zip(predictions.flatten(), labels.flatten()) if l != -100]
true_labels = [l for l in labels.flatten() if l != -100]


# PRINT CLASSIFICATION REPORT
print("\n" + "="*55)
print(f"CLASSIFICATION REPORT {model_path}")
print("="*55)
report = classification_report(
    true_labels, 
    true_predictions, 
    target_names=["Word (0)", "Sentence Boundary (1)"], 
    digits=4
)
print(report)


# PLOT CONFUSION MATRIX
print("\nGenerating Confusion Matrix plot...")
cm = confusion_matrix(true_labels, true_predictions)

plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=["Predicted: Word", "Predicted: Boundary"], 
                 yticklabels=["Actual: Word", "Actual: Boundary"],
                 annot_kws={"size": 14}) 

plt.title('Confusion Matrix - SBD XLM-RoBERTa', fontsize=16, pad=20)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plot_path = "eval_results/confusion_matrix_sbd(testset).png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to: {plot_path}")
