import numpy as np
from datasets import Value, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = "./models/cooking-safety-classifier"

LABEL2ID = {"SAFE_COOKING": 0, "HARMFUL": 1, "NON_COOKING": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode_label(example):
    if example["label"] not in LABEL2ID:
        raise ValueError(f"Unknown label: {example['label']}")
    example["label"] = LABEL2ID[example["label"]]
    return example


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/train.csv",
            "validation": "data/validation.csv",
            "test": "data/test.csv",
        },
    )

    dataset = dataset.map(encode_label)
    dataset = dataset.cast_column("label", Value("int64"))
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Validation metrics ===")
    print(trainer.evaluate())

    print("\n=== Test set evaluation ===")
    predictions = trainer.predict(dataset["test"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)

    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    print(
        classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(target_names)
    print(confusion_matrix(y_true, y_pred))

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
