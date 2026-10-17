import torch
import torch.nn as nn

class MNIST_MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128, 64], num_classes=10):
        super(MNIST_MLP, self).__init__()
        
        # Giriş matrisini (Batch, 1, 28, 28) -> (Batch, 784) yapar
        self.flatten = nn.Flatten()
        
        # Katmanların tanımlanması
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], num_classes)
            # Not: PyTorch'un CrossEntropyLoss fonksiyonu içerisinde Softmax dahili
            # olarak hesaplandığı için son katmana Softmax eklenmez (logits üretilir).
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits