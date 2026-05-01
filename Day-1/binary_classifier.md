# Binary Classifier with PyTorch — Complete Guide

A step-by-step walkthrough for building, training, and evaluating a binary classification neural network using PyTorch.

---

## Table of Contents

1. [Import Libraries](#step-1-import-libraries)
2. [Import the Dataframe](#step-2-import-the-dataframe)
3. [Check Class Balance](#step-3-check-class-balance)
4. [Data Cleaning & Feature Engineering](#step-4-data-cleaning--feature-engineering)
5. [Separate Train & Test Data](#step-5-separate-train--test-data)
6. [Standardize the Data](#step-6-standardize-the-data)
7. [Convert to NumPy Arrays](#step-7-convert-to-numpy-arrays)
8. [Separate Features from Target](#step-8-separate-features-from-target)
9. [Define the Model](#step-9-define-the-model)
10. [Training Setup](#step-10-training-setup)
11. [Training Loop](#step-11-training-loop)
12. [Validation & Evaluation](#step-12-validation--evaluation)

---

## Step 1: Import Libraries

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Check if GPU is available (e.g., in Google Colab)
print(torch.cuda.is_available())
```

---

## Step 2: Import the Dataframe

Pandas is used to read the CSV file and create a dataframe instance.

```python
df = pd.read_csv("link")
df.shape   # returns the dimensions of the dataframe
df.head()  # returns first 5 rows
```

---

## Step 3: Check Class Balance

Check the balance between 0s and 1s in the target column.

```python
df.target.value_counts(normalize=True, dropna=False)
```

- `normalize=True` → shows proportions (0 ≤ x ≤ 1)
- `dropna=False` → includes NaN values
- Selects the target column from `df`

> A balanced dataset is important to avoid biased model training.

---

## Step 4: Data Cleaning & Feature Engineering

```python
df = pd.read_csv("link")

categorical_variables = ['...', '...']   # non-integer variables
numerics = ['...', '...']                # numeric/number values

# One-hot encode categorical variables (convert features to numbers)
df = pd.get_dummies(df, columns=categorical_variables, dtype=np.int64)
```

---

## Step 5: Separate Train & Test Data

```python
# Sample 20% of data for testing
test_df = df.sample(frac=0.2, random_state=42)

# Use remaining 80% for training
train_df = df.drop(test_df.index)
```

**Notes:**
- `df.sample()` — picks random rows from the dataframe
- `random_state` — assigns a value to the randomness so results can be reproduced
- `train_df.drop()` — drops all test data; this set is used exclusively for training the model

---

## Step 6: Standardize the Data

Neural networks work much better when all features are on the same scale. Since the neural network performs repeated weighted sums (`z = w₁x₁ + w₂x₂ + ...`), if one feature is large while others are very small, the large feature dominates the sum.

**Standardization formula (z-score normalization):**

```
x' = (x - mean) / σ
```

Applied for each feature in the dataframe:

```python
# Define mean and standard deviation from training data
mean = train_df[numerics].mean()
sd   = train_df[numerics].std()

# Standardize both train and test sets
train_df[numerics] = (train_df[numerics] - mean) / sd
test_df[numerics]  = (test_df[numerics] - mean) / sd

train_df.head()
```

> ⚠️ Always compute `mean` and `sd` from **training data only**, then apply to both train and test. This prevents data leakage.

---

## Step 7: Convert to NumPy Arrays

```python
train = train_df.to_numpy()
test  = test_df.to_numpy()
```

---

## Step 8: Separate Features from Target (Y)

```python
# Features: all columns except column 6
train_X = np.delete(train, 6, axis=1)
test_X  = np.delete(test,  6, axis=1)

# Target: column 6 only
train_y = train[:, 6]
test_y  = test[:, 6]
```

---

## Step 9: Define the Model

Create a custom neural network class by subclassing `nn.Module`.

```python
class BinaryClassifier(nn.Module):   # creating your own model; inherits from nn.Module

    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()

        self.hidden  = nn.Linear(input_dim, 16)  # input → 16 neurons (hidden layer)
        self.relu    = nn.ReLU()
        self.output  = nn.Linear(16, 1)           # 16 neurons → 1 output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):   # defines how data flows through the network
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

# Data flow: Input → Hidden Layer → ReLU → Output Layer → Sigmoid → Output

model = BinaryClassifier(train_X.shape[0])   # shape of training dataframe gives input_dim
print(model)
```

---

## Step 10: Training Setup

```python
from sklearn.model_selection import train_test_split   # optional utility
import torch.optim as optim

# Loss function — Binary Cross Entropy (for binary classification)
criterion = nn.BCELoss()

# Optimizer — Adam (defines how the model learns)
# lr = learning rate; updates all weights/biases
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert dataframes to tensors
# float32 is required for BCELoss
# .view(-1, 1) converts row vector to column vector
X      = torch.tensor(train_X.values, dtype=torch.float32)
y      = torch.tensor(train_y.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(test_X.values,  dtype=torch.float32)
y_test = torch.tensor(test_y.values,  dtype=torch.float32).view(-1, 1)

epochs    = 200   # number of full passes over the training data
batch_size = 32   # samples per mini-batch (for mini-batch gradient descent / SGD)

# Dictionary to store metrics per epoch
history = {
    "loss":         [],
    "accuracy":     [],
    "val-loss":     [],
    "val-accuracy": []
}
```

---

## Step 11: Training Loop

Finally, training starts! 🎉

```python
for epoch in range(epochs):           # outer loop: runs 200 times

    model.train()                     # set model to training mode

    permutation = torch.randperm(X.size(0))   # shuffle data each epoch
    epoch_loss  = 0
    correct     = 0                   # number of correct predictions

    for p in range(0, X.size(0), batch_size):   # inner loop for mini-batches

        indices = permutation[p : p + batch_size]   # random indices
        batch_X = X[indices]
        batch_y = y[indices]

        optimizer.zero_grad()                        # clear old gradients

        outputs = model(batch_X)                     # forward pass: input data to model
        loss    = criterion(outputs, batch_y)        # compute loss (prediction vs truth)
        loss.backward()                              # backward pass (chain rule)
        optimizer.step()                             # update weights

        epoch_loss += loss.item()                    # accumulate loss

        preds   = (outputs >= 0.5).float()           # how many predictions are right
        correct += (preds == batch_y).sum().item()

    train_acc = correct / len(y_train)              # accuracy formula

    # Append metrics to history
    history["loss"].append(epoch_loss)
    history["accuracy"].append(train_acc)
```

---

## Step 12: Validation & Evaluation

```python
    # Evaluate on validation/test set
    model.eval()                       # set model to evaluation mode
    with torch.no_grad():              # no gradients needed for inference

        val_outputs = model(X_test)
        val_loss    = criterion(val_outputs, y_test).item()
        val_preds   = (val_outputs >= 0.5).float()
        val_acc     = (val_preds == y_test).sum().item() / len(y_test)

    # Append validation metrics
    history["val-loss"].append(val_loss)
    history["val-accuracy"].append(val_acc)

    # Print progress each epoch
    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"loss={epoch_loss:.4f} "
        f"acc={train_acc:.4f} "
        f"val-loss={val_loss:.4f} "
        f"val-acc={val_acc:.4f}"
    )

# We may also plot these metrics to check for overfitting
```

---

## Plotting Training History (Optional)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(history["loss"],     label="Train Loss")
axes[0].plot(history["val-loss"], label="Val Loss")
axes[0].set_title("Loss over Epochs")
axes[0].legend()

# Accuracy curve
axes[1].plot(history["accuracy"],     label="Train Accuracy")
axes[1].plot(history["val-accuracy"], label="Val Accuracy")
axes[1].set_title("Accuracy over Epochs")
axes[1].legend()

plt.tight_layout()
plt.show()
```

> Plot both train and validation curves to detect **overfitting** — if train accuracy is high but val accuracy is low, the model is memorizing rather than generalizing.

---

## Key Concepts Summary

| Concept | Description |
|---|---|
| `nn.Module` | Base class for all PyTorch models |
| `nn.BCELoss` | Binary Cross Entropy loss for binary classification |
| `optim.Adam` | Adaptive optimizer; updates weights/biases during training |
| `nn.ReLU` | Activation function; introduces non-linearity |
| `nn.Sigmoid` | Squashes output to [0, 1]; used for binary probability |
| Mini-batch SGD | Train on small batches (32) instead of full dataset at once |
| `model.train()` | Enables dropout/batch norm during training |
| `model.eval()` | Disables dropout/batch norm during evaluation |
| `torch.no_grad()` | Disables gradient computation (faster inference) |
| Standardization | Scale features to same range to improve convergence |
| `random_state` | Seed for reproducibility |
| One-hot encoding | Convert categorical variables to numeric 0/1 columns |