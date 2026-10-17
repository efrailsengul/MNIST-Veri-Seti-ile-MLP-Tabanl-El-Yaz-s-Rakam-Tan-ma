import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(batch_size=64, val_size=10000):
    # MNIST piksel değerleri [0, 255] arasındadır.
    # ToTensor() bunları [0.0, 1.0] aralığına çeker.
    # Normalize((0.1307,), (0.3081,)) MNIST'in global ortalama ve standart sapmasıdır.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 1. Ham MNIST eğitim ve test setlerini indir
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 2. 60.000'lik seti Train (50.000) ve Validation (10.000) olarak ayır
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # Tekrarlanabilirlik için seed
    )

    # 3. DataLoader'ları hazırla
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader