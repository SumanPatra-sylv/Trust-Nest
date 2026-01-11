"""
DistilBERT Scam Classifier Training
====================================
Trains distilbert-base-uncased on SMS + WhatsApp datasets.

Features:
- Class weighting for scam-heavy imbalance
- Precision/Recall/F1 logging
- Saves to models/distilbert/
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] Model: {MODEL_NAME}")
print(f"[CONFIG] Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_dir: str) -> pd.DataFrame:
    """Load SMS and WhatsApp data."""
    sms_path = os.path.join(data_dir, "public_sms.csv")
    wa_path = os.path.join(data_dir, "public_whatsapp.csv")
    
    dfs = []
    
    if os.path.exists(sms_path):
        sms = pd.read_csv(sms_path)
        sms["text"] = sms["message_text"]
        sms["modality"] = "SMS"
        dfs.append(sms)
        print(f"[DATA] SMS: {len(sms)} samples")
    
    if os.path.exists(wa_path):
        wa = pd.read_csv(wa_path)
        wa["text"] = wa["conversation_text"]
        wa["modality"] = "WhatsApp"
        dfs.append(wa)
        print(f"[DATA] WhatsApp: {len(wa)} samples")
    
    data = pd.concat(dfs, ignore_index=True)
    data["text"] = data["text"].fillna("").astype(str)
    
    # Binary classification: is_scam
    data["label"] = data["is_scam"].astype(int)
    
    print(f"[DATA] Total: {len(data)} samples")
    print(f"[DATA] Scams: {data['label'].sum()} ({data['label'].mean():.1%})")
    print(f"[DATA] Legit: {(data['label'] == 0).sum()} ({(data['label'] == 0).mean():.1%})")
    
    return data


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class ScamDataset(Dataset):
    """PyTorch Dataset for scam classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# CUSTOM TRAINER WITH CLASS WEIGHTS
# ============================================================================

class WeightedTrainer(Trainer):
    """Trainer with class weighting for imbalanced data."""
    
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Weighted cross-entropy
        weight = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    """Compute precision, recall, F1."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_distilbert(data_dir: str, output_dir: str):
    """Train DistilBERT on scam data."""
    
    print("\n" + "=" * 70)
    print("DISTILBERT SCAM CLASSIFIER TRAINING")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] LOADING DATA")
    data = load_data(data_dir)
    
    texts = data["text"].values
    labels = data["label"].values
    
    # Split data
    print("\n[2/5] SPLITTING DATA")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )
    print(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Compute class weights for imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )
    print(f"[WEIGHTS] Class 0 (Legit): {class_weights[0]:.2f}")
    print(f"[WEIGHTS] Class 1 (Scam): {class_weights[1]:.2f}")
    
    # Load tokenizer
    print("\n[3/5] LOADING MODEL")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "SAFE", 1: "SCAM"},
        label2id={"SAFE": 0, "SCAM": 1}
    )
    model.to(DEVICE)
    print(f"[MODEL] Loaded {MODEL_NAME}")
    print(f"[MODEL] Parameters: {model.num_parameters():,}")
    
    # Create datasets
    train_dataset = ScamDataset(X_train, y_train, tokenizer)
    test_dataset = ScamDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        report_to="none"  # Disable wandb/tensorboard
    )
    
    # Trainer with class weights
    print("\n[4/5] TRAINING")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    train_result = trainer.train()
    
    # Evaluate
    print("\n[5/5] EVALUATION")
    eval_result = trainer.evaluate()
    
    print("\n" + "=" * 70)
    print("FINAL METRICS (Test Set)")
    print("=" * 70)
    print(f"  Accuracy:  {eval_result['eval_accuracy']:.4f}")
    print(f"  Precision: {eval_result['eval_precision']:.4f}")
    print(f"  Recall:    {eval_result['eval_recall']:.4f}")
    print(f"  F1 Score:  {eval_result['eval_f1']:.4f}")
    
    # Detailed classification report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)
    
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    
    print(classification_report(y_test, y_pred, target_names=["SAFE", "SCAM"]))
    
    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    model_save_path = os.path.join(output_dir, "distilbert")
    os.makedirs(model_save_path, exist_ok=True)
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save training metadata
    metadata = {
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_weights": class_weights.tolist(),
        "metrics": {
            "accuracy": float(eval_result["eval_accuracy"]),
            "precision": float(eval_result["eval_precision"]),
            "recall": float(eval_result["eval_recall"]),
            "f1": float(eval_result["eval_f1"])
        }
    }
    
    with open(os.path.join(model_save_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[SAVE] Model saved to: {model_save_path}")
    print(f"[SAVE] Files: config.json, model.safetensors, tokenizer.json, training_metadata.json")
    
    # List saved files
    saved_files = os.listdir(model_save_path)
    for f in saved_files:
        size = os.path.getsize(os.path.join(model_save_path, f))
        print(f"       - {f} ({size / 1024:.1f} KB)")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return eval_result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, "models")
    
    train_distilbert(project_dir, model_dir)
