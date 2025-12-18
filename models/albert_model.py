import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


class NERDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'],
            'attention_mask': self.inputs[idx]['attention_mask'],
            'labels': self.labels[idx]
        }


class ALBERTForNER(nn.Module):
    def __init__(self, num_labels, dropout=0.1):
        super(ALBERTForNER, self).__init__()
        self.num_labels = num_labels

        # Load pre-trained ALBERT model
        self.albert = AlbertModel.from_pretrained('albert-base-v2')

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.albert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Get ALBERT outputs
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=11)  # PAD token index
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}


class ALBERTNERTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def train(self, train_inputs, train_labels, epochs=5, batch_size=16, learning_rate=2e-5):
        """Train the ALBERT NER model"""
        # Create dataset and dataloader
        train_dataset = NERDataset(train_inputs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs['loss']
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

    def predict(self, test_inputs):
        """Make predictions on test data"""
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for inp in test_inputs:
                input_ids = inp['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = inp['attention_mask'].unsqueeze(0).to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.append(predictions.squeeze(0).cpu().tolist())

        return all_predictions

    def save_model(self, save_path):
        """Save model to disk"""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pt'))
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Load model from disk"""
        model_path = os.path.join(load_path, 'model.pt')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {load_path}")
        else:
            print(f"No model found at {load_path}")