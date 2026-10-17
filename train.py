import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_data_loaders
from model import MNIST_MLP

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim Cihazı: {device}")

    # Hiperparametreler
    epochs = 15
    batch_size = 64
    learning_rate = 0.001

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)
    model = MNIST_MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {t_loss:.4f} - Train Acc: %{t_acc*100:.2f} | "
              f"Val Loss: {v_loss:.4f} - Val Acc: %{v_acc*100:.2f}")

    total_training_time = time.time() - start_time
    print(f"\nToplam Eğitim Süresi: {total_training_time:.2f} saniye")

    # Nihai Test Doğrulaması (Test verisi eğitimde hiç kullanılmadı)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"--> Nihai Test Loss: {test_loss:.4f} | Nihai Test Accuracy: %{test_acc*100:.2f}")

    # Ağırlıkları kaydet
    torch.save(model.state_dict(), "mnist_mlp_weights.pth")

    # Grafikleri Çiz ve Kaydet
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', marker='s')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history['train_acc']], label='Train Acc', marker='o')
    plt.plot(range(1, epochs + 1), [a * 100 for a in history['val_acc']], label='Val Acc', marker='s')
    plt.title('Training & Validation Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("plots/training_curves.png", dpi=300)
    print("Grafik 'plots/training_curves.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()