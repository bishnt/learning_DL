# MNIST Digit Classifier with PyTorch

A step-by-step walkthrough for building a multi-class neural network to classify handwritten digits (0–9) using the MNIST dataset and PyTorch.

---

## Table of Contents

1. [Import Libraries](#1-import-libraries)
2. [Load & Explore MNIST](#2-load--explore-mnist)
3. [One-Hot Encoding](#3-one-hot-encoding)
4. [Custom Dataset Class](#4-custom-dataset-class)
5. [DataLoader](#5-dataloader)
6. [Define the Model](#6-define-the-model)
7. [Training Loop](#7-training-loop)
8. [Plot Training Loss](#8-plot-training-loss)
9. [Evaluation & Visualization](#9-evaluation--visualization)
10. [Key Concepts](#key-concepts)

---

## 1. Import Libraries

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

# Check if GPU is available
torch.cuda.is_available()
```

For Google Colab, mount your Drive to access saved datasets:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 2. Load & Explore MNIST

The MNIST dataset contains 60,000 training and 10,000 test grayscale images of handwritten digits, each 28×28 pixels.

```python
mnist = datasets.MNIST(root='./data', train=True, download=True)

x = torch.tensor(mnist.data,    dtype=torch.float32)  # shape: (60000, 28, 28)
y = torch.tensor(mnist.targets, dtype=torch.long)      # shape: (60000,)
```

Visualize a sample:

```python
plt.imshow(x[4].numpy())
plt.title(f'Number: {y[4].numpy()}')
plt.colorbar()
plt.show()
```

Flatten images for the linear network (28×28 = 784 features):

```python
x.view(-1, 28*28).shape   # → (60000, 784)
```

---

## 3. One-Hot Encoding

For multi-class classification the targets need to be one-hot encoded — a vector of length 10 with a `1` at the correct digit index and `0` elsewhere.

```python
y_encoded = F.one_hot(y, num_classes=10)
y_encoded.shape   # → (60000, 10)
```

Example: digit `3` → `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`

---

## 4. Custom Dataset Class

A custom `Dataset` wraps the data and handles normalization and encoding automatically.

```python
class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 225                             # normalize pixel values to [0, 1]
        self.y = F.one_hot(self.y, num_classes=10).to(float)   # one-hot encode labels

    def __len__(self):
        return self.x.shape[0]                            # total number of samples

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]                     # return one sample by index
```

Load train and test datasets:

```python
train_ds = CTDataset("/content/drive/MyDrive/.../MNIST/processed/training.pt")
test_ds  = CTDataset("/content/drive/MyDrive/.../MNIST/processed/test.pt")

len(train_ds)   # 60000
```

> **Why normalize?** Dividing by 225 brings pixel values from [0, 255] into [0, 1], which helps gradient descent converge faster and more stably.

---

## 5. DataLoader

`DataLoader` wraps the dataset and handles mini-batching, shuffling, and parallel loading.

```python
train_dl = DataLoader(train_ds, batch_size=5)

# Verify shape of one batch
for x, y in train_dl:
    print(x.shape)   # (5, 28, 28)
    print(y.shape)   # (5, 10)
    break
```

---

## 6. Define the Model

A 3-layer fully connected (dense) neural network:

```
Input (784) → Hidden Layer 1 (100) → ReLU
            → Hidden Layer 2 (50)  → ReLU
            → Output Layer  (10)   → raw logits
```

```python
class digitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 100)   # 784 → 100
        self.layer2 = nn.Linear(100,   50)    # 100 → 50
        self.layer3 = nn.Linear(50,    10)    # 50  → 10 (one per digit class)
        self.R      = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)          # flatten: (batch, 28, 28) → (batch, 784)
        x = self.R(self.layer1(x))     # hidden layer 1 + ReLU
        x = self.R(self.layer2(x))     # hidden layer 2 + ReLU
        x = self.layer3(x)             # output logits (no activation — CrossEntropyLoss handles it)
        return x.squeeze()

model = digitClassifier()
```

> **Why no Sigmoid/Softmax at the end?** PyTorch's `nn.CrossEntropyLoss` internally applies `log_softmax`, so the model outputs raw logits and the loss function handles the final activation.

---

## 7. Training Loop

```python
def train_model(dl, f, n_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(f.parameters(), lr=0.001)

    loss_log   = []
    epoch_log  = []
    N = len(dl)   # number of batches per epoch

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch}')

        for i, (x, y) in enumerate(dl):
            optimizer.zero_grad()              # clear old gradients

            loss_value = criterion(f(x), y)   # forward pass + compute loss
            loss_value.backward()              # backward pass (compute gradients)
            optimizer.step()                   # update weights

            epoch_log.append(epoch + i / N)   # fractional epoch position
            loss_log.append(loss_value.item())

    return np.array(epoch_log), np.array(loss_log)

epoch_data, loss_data = train_model(train_dl, model)
```

**What happens each iteration:**
1. `zero_grad()` — clears accumulated gradients from the previous step
2. Forward pass — input → model → raw logits
3. `criterion(...)` — computes cross-entropy loss between logits and one-hot targets
4. `backward()` — computes gradients via backpropagation (chain rule)
5. `optimizer.step()` — updates all weights using Adam

---

## 8. Plot Training Loss

```python
# Raw loss per batch
plt.plot(epoch_data, loss_data)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Smoothed: average loss per epoch
plt.plot(
    epoch_data.reshape(20, -1).mean(axis=1),
    loss_data.reshape(20, -1).mean(axis=1)
)
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.title('Cross Entropy Average')
plt.show()
```

A decreasing loss curve confirms the model is learning. The smoothed average (per epoch mean) removes batch-level noise for clearer visualization.

---

## 9. Evaluation & Visualization

Run inference on 2000 test samples and check predictions:

```python
xs, ys = test_ds[:2000]
yhats = model(xs).argmax(axis=1)   # pick the class with the highest logit score
```

> `.argmax(axis=1)` converts the 10-dimensional output vector into a single predicted digit (0–9).

Visualize a grid of predictions:

```python
figure, ax = plt.subplots(10, 4, figsize=(10, 15))

for i in range(40):
    plt.subplot(10, 4, i + 1)
    plt.imshow(xs[i])
    plt.title(f'Predicted Digit: {yhats[i]}')

figure.tight_layout()
plt.show()
```

---

## Key Concepts

| Concept | Description |
|---|---|
| `Dataset` | Abstract class; must implement `__len__` and `__getitem__` |
| `DataLoader` | Wraps a Dataset; handles batching, shuffling, and multi-threading |
| Normalization (`/ 225`) | Scales pixel values to [0, 1] for stable training |
| One-hot encoding | Converts integer labels to binary vectors; needed for `CrossEntropyLoss` |
| `nn.CrossEntropyLoss` | Combines `log_softmax` + `NLLLoss`; expects raw logits, not probabilities |
| `optimizer.zero_grad()` | Must be called before each backward pass to clear stale gradients |
| `.argmax(axis=1)` | Converts logits → predicted class index during inference |
| `x.view(-1, 28*28)` | Flattens a 2D image tensor into a 1D vector for the linear layer |
| Adam optimizer | Adaptive learning rate optimizer; generally converges faster than plain SGD |
| ReLU | Activation function; adds non-linearity; prevents vanishing gradients |

---

## Architecture Summary

```
Input Image (28×28 = 784 pixels)
        ↓
  Linear(784 → 100) + ReLU
        ↓
  Linear(100 → 50)  + ReLU
        ↓
  Linear(50  → 10)       ← raw logits (one per digit)
        ↓
  CrossEntropyLoss        ← applies softmax internally during training
        ↓
  argmax()               ← final predicted digit during inference
```