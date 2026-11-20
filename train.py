"""Train the digit CNN on MNIST and save weights to digit_cnn.pt."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN

EPOCHS = 6
device = "mps" if torch.backends.mps.is_available() else "cpu"

norm = transforms.Normalize((0.1307,), (0.3081,))
train_tf = transforms.Compose([
    # augmentation makes the model more tolerant of hand-drawn variation
    transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=8),
    transforms.ToTensor(), norm,
])
test_tf = transforms.Compose([transforms.ToTensor(), norm])

train_ds = datasets.MNIST("data", train=True, download=True, transform=train_tf)
test_ds = datasets.MNIST("data", train=False, download=True, transform=test_tf)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=1000)

model = DigitCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    print(f"epoch {epoch}/{EPOCHS}  test accuracy: {correct / len(test_ds):.4f}", flush=True)

torch.save(model.cpu().state_dict(), "digit_cnn.pt")
print("saved digit_cnn.pt")
