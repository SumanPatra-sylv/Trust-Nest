"""
Retrain DistilBERT on larger combined dataset
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Load and combine datasets
print("Loading datasets...")

all_data = []

# Load unified multimodal (main dataset)
try:
    df = pd.read_csv('public_unified_multimodal.csv')
    if 'text' in df.columns and 'is_scam' in df.columns:
        for _, row in df.iterrows():
            all_data.append({
                'text': str(row['text']),
                'label': int(row['is_scam'])
            })
        print(f"  public_unified_multimodal.csv: {len(df)} samples")
except Exception as e:
    print(f"  Error loading unified: {e}")

# Load SMS
try:
    df = pd.read_csv('public_sms.csv')
    text_col = 'text' if 'text' in df.columns else 'content' if 'content' in df.columns else df.columns[-1]
    label_col = 'is_scam' if 'is_scam' in df.columns else 'label' if 'label' in df.columns else None
    if label_col:
        for _, row in df.iterrows():
            all_data.append({
                'text': str(row[text_col]),
                'label': int(row[label_col])
            })
        print(f"  public_sms.csv: {len(df)} samples")
except Exception as e:
    print(f"  Error loading SMS: {e}")

# Load WhatsApp
try:
    df = pd.read_csv('public_whatsapp.csv')
    text_col = 'text' if 'text' in df.columns else 'content' if 'content' in df.columns else df.columns[-1]
    label_col = 'is_scam' if 'is_scam' in df.columns else 'label' if 'label' in df.columns else None
    if label_col:
        for _, row in df.iterrows():
            all_data.append({
                'text': str(row[text_col]),
                'label': int(row[label_col])
            })
        print(f"  public_whatsapp.csv: {len(df)} samples")
except Exception as e:
    print(f"  Error loading WhatsApp: {e}")

# Add more legitimate messages to balance
legitimate_examples = [
    "Good morning! How are you?",
    "Meeting at 3pm tomorrow",
    "Happy birthday!",
    "Your order has been delivered",
    "See you at dinner",
    "Call me when you're free",
    "The weather is nice today",
    "Thanks for your help",
    "I'll be there in 10 minutes",
    "Don't forget the groceries",
    "Movie starts at 7",
    "Your appointment is confirmed",
    "Package delivered successfully",
    "Flight lands at 5pm",
    "Dinner is ready",
    "I reached home safely",
    "Good night",
    "Take care",
    "Have a nice day",
    "See you soon",
    "Your Amazon order shipped",
    "Your Flipkart order delivered",
    "Train PNR confirmed",
    "Bus ticket booked",
    "Hotel booking confirmed",
    "Your refund processed",
    "Payment successful",
    "Balance: Rs 5000",
    "EMI reminder for tomorrow",
    "School closed tomorrow",
]

for text in legitimate_examples:
    all_data.append({'text': text, 'label': 0})

print(f"\nTotal samples: {len(all_data)}")

# Create DataFrame
df = pd.DataFrame(all_data)
df = df.drop_duplicates(subset=['text'])
print(f"After dedup: {len(df)}")

# Check balance
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), 
    test_size=0.15, random_state=42, stratify=df['label']
)

print(f"\nTrain: {len(train_texts)}, Test: {len(test_texts)}")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
print(f"Class weights: SAFE={class_weights[0]:.2f}, SCAM={class_weights[1]:.2f}")

# PyTorch Dataset
class ScamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialize
print("\nLoading DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Device: {device}")

# Create datasets
train_dataset = ScamDataset(train_texts, train_labels, tokenizer)
test_dataset = ScamDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Training
optimizer = AdamW(model.parameters(), lr=2e-5)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

EPOCHS = 3
print(f"\nTraining for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Evaluation
print("\nEvaluating...")
model.eval()
correct = 0
total = 0
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.2%}")

# Confusion matrix
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=['SAFE', 'SCAM']))

# Save model
print("\nSaving model...")
os.makedirs('models/distilbert', exist_ok=True)
model.save_pretrained('models/distilbert')
tokenizer.save_pretrained('models/distilbert')

print("\n✅ Model retrained and saved to models/distilbert/")
print("Restart the backend to use the new model.")
