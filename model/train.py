import torch
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from model.config import DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE
from model.model import SentimentClassifierBert

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    all_epoch_preds, all_epoch_labels = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        total_samples = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for i, batch in enumerate(train_loader_tqdm):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += len(batch["input_ids"])
            train_loader_tqdm.set_postfix({"loss": f"{loss.item():.4f}",
                                            "acc": f"{(outputs.argmax(dim=1) == labels).float().mean().item() * 100:.2f}%"})

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        train_loss.append(avg_loss)
        train_acc.append(avg_acc)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Training Accuracy: {avg_acc * 100:.2f}%")

        model.eval()
        total_test_loss, total_test_correct = 0, 0
        total_test_samples = 0
        all_preds = []
        all_labels = []

        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        with torch.no_grad():
            for i, batch in enumerate(test_loader_tqdm):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                total_test_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_test_samples += len(batch["input_ids"])
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                test_loader_tqdm.set_postfix({"loss": f"{loss.item():.4f}",
                                               "acc": f"{(outputs.argmax(dim=1) == labels).float().mean().item() * 100:.2f}%"})

        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_acc = total_test_correct / total_test_samples
        test_loss.append(avg_test_loss)
        test_acc.append(avg_test_acc)
        all_epoch_preds.extend(all_preds)
        all_epoch_labels.extend(all_labels)

        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc * 100:.2f}%")

    return train_loss, test_loss, train_acc, test_acc
