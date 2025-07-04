import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

def train_one_epoch_classifier(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x_batch, _, p_batch = batch
        x_batch = x_batch.to(device)
        p_batch = p_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, p_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += p_batch.size(0)
        correct += (predicted == p_batch).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def test_one_epoch_classifier(model, dataloader, criterion, device, predictions=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    if predictions:
        all_predictions = []
        all_labels = []
        all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            x_batch, _, p_batch = batch
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, p_batch)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += p_batch.size(0)
            correct += (predicted == p_batch).sum().item()
            if predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(p_batch.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    if predictions:
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        return total_loss / len(dataloader), correct / total, all_predictions, all_labels, all_probs
    
    else:
        return total_loss / len(dataloader), correct / total
    
def train_classifier(config, model, train_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    best_accuracy = 0.0
    best_model = None
    for epoch in range(config['num_epochs']):
        train_loss, train_accuracy = train_one_epoch_classifier(model, train_dataloader, optimizer, criterion, device)
        test_loss, test_accuracy = test_one_epoch_classifier(model, test_dataloader, criterion, device)

        print(f"Epoch [{epoch+1}/{config['num_epochs']}]: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model.state_dict()
            torch.save(best_model, config['model_save_path'])
    return best_model

def evaluate_classifier(config, model, test_dataset, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(config['model_save_path']))

    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loss, test_accuracy, predictions, labels, probs = test_one_epoch_classifier(model, test_dataloader, criterion, device, predictions=True)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy, predictions, labels, probs
