\# Binary Classifier with PyTorch



A step-by-step walkthrough for building, training, and evaluating a binary classification neural network using PyTorch.



\---



\## Table of Contents



1\. \[Import Libraries](#step-1-import-libraries)

2\. \[Import the Dataframe](#step-2-import-the-dataframe)

3\. \[Check Class Balance](#step-3-check-class-balance)

4\. \[Data Cleaning \& Feature Engineering](#step-4-data-cleaning--feature-engineering)

5\. \[Separate Train \& Test Data](#step-5-separate-train--test-data)

6\. \[Standardize the Data](#step-6-standardize-the-data)

7\. \[Convert to NumPy Arrays](#step-7-convert-to-numpy-arrays)

8\. \[Separate Features from Target](#step-8-separate-features-from-target)

9\. \[Define the Model](#step-9-define-the-model)

10\. \[Training Setup](#step-10-training-setup)

11\. \[Training Loop](#step-11-training-loop)

12\. \[Validation \& Evaluation](#step-12-validation--evaluation)



\---



\## Step 1: Import Libraries



```python

import torch

import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



\# Check if GPU is available (e.g., in Google Colab)

print(torch.cuda.is\_available())

```



\---



\## Step 2: Import the Dataframe



Pandas is used to read the CSV file and create a dataframe instance.



```python

df = pd.read\_csv("link")

df.shape   # returns the dimensions of the dataframe

df.head()  # returns first 5 rows

```



\---



\## Step 3: Check Class Balance



Check the balance between 0s and 1s in the target column.



```python

df.target.value\_counts(normalize=True, dropna=False)

```



\- `normalize=True` → shows proportions (0 ≤ x ≤ 1)

\- `dropna=False` → includes NaN values

\- Selects the target column from `df`



> A balanced dataset is important to avoid biased model training.



\---



\## Step 4: Data Cleaning \& Feature Engineering



```python

df = pd.read\_csv("link")



categorical\_variables = \['...', '...']   # non-integer variables

numerics = \['...', '...']                # numeric/number values



\# One-hot encode categorical variables (convert features to numbers)

df = pd.get\_dummies(df, columns=categorical\_variables, dtype=np.int64)

```



\---



\## Step 5: Separate Train \& Test Data



```python

\# Sample 20% of data for testing

test\_df = df.sample(frac=0.2, random\_state=42)



\# Use remaining 80% for training

train\_df = df.drop(test\_df.index)

```



\*\*Notes:\*\*

\- `df.sample()` — picks random rows from the dataframe

\- `random\_state` — assigns a value to the randomness so results can be reproduced

\- `train\_df.drop()` — drops all test data; this set is used exclusively for training the model



\---



\## Step 6: Standardize the Data



Neural networks work much better when all features are on the same scale. Since the neural network performs repeated weighted sums (`z = w₁x₁ + w₂x₂ + ...`), if one feature is large while others are very small, the large feature dominates the sum.



\*\*Standardization formula (z-score normalization):\*\*



```

x' = (x - mean) / σ

```



Applied for each feature in the dataframe:



```python

\# Define mean and standard deviation from training data

mean = train\_df\[numerics].mean()

sd   = train\_df\[numerics].std()



\# Standardize both train and test sets

train\_df\[numerics] = (train\_df\[numerics] - mean) / sd

test\_df\[numerics]  = (test\_df\[numerics] - mean) / sd



train\_df.head()

```



> ⚠️ Always compute `mean` and `sd` from \*\*training data only\*\*, then apply to both train and test. This prevents data leakage.



\---



\## Step 7: Convert to NumPy Arrays



```python

train = train\_df.to\_numpy()

test  = test\_df.to\_numpy()

```



\---



\## Step 8: Separate Features from Target (Y)



```python

\# Features: all columns except column 6

train\_X = np.delete(train, 6, axis=1)

test\_X  = np.delete(test,  6, axis=1)



\# Target: column 6 only

train\_y = train\[:, 6]

test\_y  = test\[:, 6]

```



\---



\## Step 9: Define the Model



Create a custom neural network class by subclassing `nn.Module`.



```python

class BinaryClassifier(nn.Module):   # creating your own model; inherits from nn.Module



&#x20;   def \_\_init\_\_(self, input\_dim):

&#x20;       super(BinaryClassifier, self).\_\_init\_\_()



&#x20;       self.hidden  = nn.Linear(input\_dim, 16)  # input → 16 neurons (hidden layer)

&#x20;       self.relu    = nn.ReLU()

&#x20;       self.output  = nn.Linear(16, 1)           # 16 neurons → 1 output

&#x20;       self.sigmoid = nn.Sigmoid()



&#x20;   def forward(self, x):   # defines how data flows through the network

&#x20;       x = self.hidden(x)

&#x20;       x = self.relu(x)

&#x20;       x = self.output(x)

&#x20;       x = self.sigmoid(x)

&#x20;       return x



\# Data flow: Input → Hidden Layer → ReLU → Output Layer → Sigmoid → Output



model = BinaryClassifier(train\_X.shape\[0])   # shape of training dataframe gives input\_dim

print(model)

```



\---



\## Step 10: Training Setup



```python

from sklearn.model\_selection import train\_test\_split   # optional utility

import torch.optim as optim



\# Loss function — Binary Cross Entropy (for binary classification)

criterion = nn.BCELoss()



\# Optimizer — Adam (defines how the model learns)

\# lr = learning rate; updates all weights/biases

optimizer = optim.Adam(model.parameters(), lr=0.001)



\# Convert dataframes to tensors

\# float32 is required for BCELoss

\# .view(-1, 1) converts row vector to column vector

X      = torch.tensor(train\_X.values, dtype=torch.float32)

y      = torch.tensor(train\_y.values, dtype=torch.float32).view(-1, 1)

X\_test = torch.tensor(test\_X.values,  dtype=torch.float32)

y\_test = torch.tensor(test\_y.values,  dtype=torch.float32).view(-1, 1)



epochs    = 200   # number of full passes over the training data

batch\_size = 32   # samples per mini-batch (for mini-batch gradient descent / SGD)



\# Dictionary to store metrics per epoch

history = {

&#x20;   "loss":         \[],

&#x20;   "accuracy":     \[],

&#x20;   "val-loss":     \[],

&#x20;   "val-accuracy": \[]

}

```



\---



\## Step 11: Training Loop



Finally, training starts! 🎉



```python

for epoch in range(epochs):           # outer loop: runs 200 times



&#x20;   model.train()                     # set model to training mode



&#x20;   permutation = torch.randperm(X.size(0))   # shuffle data each epoch

&#x20;   epoch\_loss  = 0

&#x20;   correct     = 0                   # number of correct predictions



&#x20;   for p in range(0, X.size(0), batch\_size):   # inner loop for mini-batches



&#x20;       indices = permutation\[p : p + batch\_size]   # random indices

&#x20;       batch\_X = X\[indices]

&#x20;       batch\_y = y\[indices]



&#x20;       optimizer.zero\_grad()                        # clear old gradients



&#x20;       outputs = model(batch\_X)                     # forward pass: input data to model

&#x20;       loss    = criterion(outputs, batch\_y)        # compute loss (prediction vs truth)

&#x20;       loss.backward()                              # backward pass (chain rule)

&#x20;       optimizer.step()                             # update weights



&#x20;       epoch\_loss += loss.item()                    # accumulate loss



&#x20;       preds   = (outputs >= 0.5).float()           # how many predictions are right

&#x20;       correct += (preds == batch\_y).sum().item()



&#x20;   train\_acc = correct / len(y\_train)              # accuracy formula



&#x20;   # Append metrics to history

&#x20;   history\["loss"].append(epoch\_loss)

&#x20;   history\["accuracy"].append(train\_acc)

```



\---



\## Step 12: Validation \& Evaluation



```python

&#x20;   # Evaluate on validation/test set

&#x20;   model.eval()                       # set model to evaluation mode

&#x20;   with torch.no\_grad():              # no gradients needed for inference



&#x20;       val\_outputs = model(X\_test)

&#x20;       val\_loss    = criterion(val\_outputs, y\_test).item()

&#x20;       val\_preds   = (val\_outputs >= 0.5).float()

&#x20;       val\_acc     = (val\_preds == y\_test).sum().item() / len(y\_test)



&#x20;   # Append validation metrics

&#x20;   history\["val-loss"].append(val\_loss)

&#x20;   history\["val-accuracy"].append(val\_acc)



&#x20;   # Print progress each epoch

&#x20;   print(

&#x20;       f"Epoch {epoch + 1}/{epochs} "

&#x20;       f"loss={epoch\_loss:.4f} "

&#x20;       f"acc={train\_acc:.4f} "

&#x20;       f"val-loss={val\_loss:.4f} "

&#x20;       f"val-acc={val\_acc:.4f}"

&#x20;   )



\# We may also plot these metrics to check for overfitting

```



\---



\## Plotting Training History (Optional)



```python

import matplotlib.pyplot as plt



fig, axes = plt.subplots(1, 2, figsize=(12, 4))



\# Loss curve

axes\[0].plot(history\["loss"],     label="Train Loss")

axes\[0].plot(history\["val-loss"], label="Val Loss")

axes\[0].set\_title("Loss over Epochs")

axes\[0].legend()



\# Accuracy curve

axes\[1].plot(history\["accuracy"],     label="Train Accuracy")

axes\[1].plot(history\["val-accuracy"], label="Val Accuracy")

axes\[1].set\_title("Accuracy over Epochs")

axes\[1].legend()



plt.tight\_layout()

plt.show()

```



> Plot both train and validation curves to detect \*\*overfitting\*\* — if train accuracy is high but val accuracy is low, the model is memorizing rather than generalizing.



\---



\## Key Concepts Summary



| Concept | Description |

|---|---|

| `nn.Module` | Base class for all PyTorch models |

| `nn.BCELoss` | Binary Cross Entropy loss for binary classification |

| `optim.Adam` | Adaptive optimizer; updates weights/biases during training |

| `nn.ReLU` | Activation function; introduces non-linearity |

| `nn.Sigmoid` | Squashes output to \[0, 1]; used for binary probability |

| Mini-batch SGD | Train on small batches (32) instead of full dataset at once |

| `model.train()` | Enables dropout/batch norm during training |

| `model.eval()` | Disables dropout/batch norm during evaluation |

| `torch.no\_grad()` | Disables gradient computation (faster inference) |

| Standardization | Scale features to same range to improve convergence |

| `random\_state` | Seed for reproducibility |

| One-hot encoding | Convert categorical variables to numeric 0/1 columns |

