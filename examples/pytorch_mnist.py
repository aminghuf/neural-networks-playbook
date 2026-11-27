import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
test_ds  = datasets.MNIST("data", train=False, download=True, transform=tfm)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size=512)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),            # raw logits: CrossEntropyLoss applies softmax itself
).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(5):
    model.train()                                  # enables dropout / batchnorm training mode
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                         # 1. forward
        loss = loss_fn(logits, yb)                 # 2. loss
        opt.zero_grad()                            # 3a. clear old gradients
        loss.backward()                            # 3b. backprop
        opt.step()                                 # 4. update

    model.eval()                                   # disables dropout
    correct = 0
    with torch.no_grad():                          # no graph needed for evaluation
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
    print(f"epoch {epoch}  last batch loss {loss.item():.3f}  test acc {correct / len(test_ds):.4f}")

torch.save(model.state_dict(), "mnist_mlp.pt")
