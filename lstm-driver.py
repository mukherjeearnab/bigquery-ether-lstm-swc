# %% [markdown]
# ## SET ENVIRONMENT VARIABLES

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from model import LSTM
from sklearn.metrics import f1_score
from torch.autograd import Variable
import torch
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

print(os.environ["CUDA_VISIBLE_DEVICES"])

# %% [markdown]
# ## Import and Prepare the DataSets

# %% [markdown]
# ### 1. Import Train & Test CSV(s)

# %%

# Set the TRAIN_SAMPLING_TECHNIQUE as follows
# SMOTE ---> 'sm'
# SMOTETomek ---> 'smt'
# ADASYN ---> 'ada'
TRAIN_SAMPLING_TECHNIQUE = 'sm'

# Set the TEST_SAMPLING_TECHNIQUE as follows
# RandomUnderSampler ---> 'rus'
# TomekLinks ---> 'tmk'
TEST_SAMPLING_TECHNIQUE = 'rus'

# Importing Training Set
train = pd.read_csv(f'./split_ds/opcode_{TRAIN_SAMPLING_TECHNIQUE}_TRAIN.csv')
train['opcode'] = train['opcode'].apply(lambda x: json.loads(x))

# Importing Testing Set
test = pd.read_csv(f'./split_ds/opcode_{TEST_SAMPLING_TECHNIQUE}_TEST.csv')
test['opcode'] = test['opcode'].apply(lambda x: json.loads(x))

train = train.iloc[:100]
print(train, test)


# %% [markdown]
# ### 2. Split the Test DataSet into Testing and Validiation DataSet

# %%
test, val = train_test_split(
    test, test_size=0.5, random_state=69, shuffle=True, stratify=test['swc_label'])

# %% [markdown]
# ### 3. Convert DataFrame(s) to Numpy N-D Array(s)

# %%


def pandas2tensor(data, X=True):
    tensor = Variable(torch.Tensor(data.tolist()))
    if X:
        return torch.reshape(tensor, (tensor.shape[0], 1, tensor.shape[1]))
    else:
        return torch.reshape(tensor, (tensor.shape[0], 1))


# %%
train_sequences = pandas2tensor(train['opcode'])
train_labels = pandas2tensor(train['swc_label'], X=False)

test_sequences = pandas2tensor(test['opcode'])
test_labels = pandas2tensor(test['swc_label'], X=False)

val_sequences = pandas2tensor(val['opcode'])
val_labels = pandas2tensor(val['swc_label'], X=False)

print("Train-Sequences", train_sequences.shape, type(train_sequences[0]))
print("Train-Labels", train_labels.shape, type(train_labels[0]))

print("Test-Sequences", test_sequences.shape, type(test_sequences[0]))
print("Test-Labels", test_labels.shape, type(test_labels[0]))

print("Validiation-Sequences", val_sequences.shape, type(val_sequences[0]))
print("Validiation-Labels", val_labels.shape, type(val_labels[0]))


# %% [markdown]
# ## Create & Evaluate the Deep-Learning Model (RNN based on LSTM architecture)

# %% [markdown]
# ### 1. Define the Hyper-Parameters

# %%
OPCODE_SEQ_LEN = 1800
EMBEDDING_DIM = 50
NUM_EPOCS = 1024
BATCH_SIZE = 128

LEARNING_RATE = 0.001  # 0.001 lr
INPUT_SIZE = 1800  # number of features
HIDDEN_SIZE = 2  # number of features in hidden state
NUM_LAYERS = 1  # number of stacked lstm layers
NUM_CLASSES = 1  # number of output classes

# %% [markdown]
# ### 1A. Import Evaluation metrics

# %%


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


def f1M(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

# %% [markdown]
# ### 2. Define the Neural Network Structure (Layers)


# %%

model = LSTM(num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
             hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
             seq_length=train_sequences.shape[1])

model.model

# %% [markdown]
# ### 3. Compile the model

# %%
model.compile(learning_rate=LEARNING_RATE)

# %% [markdown]
# ### 4. Fit and train the RNN model with Training and Validiation Data

# %%

# train_sequence = torch.tensor([train_sequences[0]])
# train_label = torch.tensor([train_labels[]])

model.fit(num_epochs=NUM_EPOCS, X=train_sequences, y=train_labels, batch_size=50,
          X_validation=val_sequences, y_validation=val_labels)


# %% [markdown]
# ### 5. Evaluate performance of model using Testing DataSet

# %%
results = model.evaluate(test_sequences, test_labels, batch_size=BATCH_SIZE)
print("Test Loss, Test Accuracy:", results)

# %% [markdown]
# ### 6. Save the model as HDF5 file

# %%
model.save(
    f'./models/model_{TRAIN_SAMPLING_TECHNIQUE}_{TEST_SAMPLING_TECHNIQUE}_{NUM_EPOCS}.h5')

# %%
# Save History as Pickle
with open(f'./models/history_{TRAIN_SAMPLING_TECHNIQUE}_{TEST_SAMPLING_TECHNIQUE}_{NUM_EPOCS}.pickle', 'wb') as fh:
    pickle.dump(history.history, fh)

# Save Results as Pickle
with open(f'./models/results_{TRAIN_SAMPLING_TECHNIQUE}_{TEST_SAMPLING_TECHNIQUE}_{NUM_EPOCS}.pickle', 'wb') as fh:
    pickle.dump(results, fh)

# %% [markdown]
# ### 7. Plot performance metrics of the Deep-Learning Model

# %%
sns.set()

# %%
# Accuracy Metrics
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

# %%
# Loss Metrics
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend(['loss', 'val_loss'])
plt.show()

# %%
pred_test_classes = model.predict_classes(
    test_sequences, verbose=1, batch_size=128)
pred_train_classes = model.predict_classes(
    train_sequences, verbose=1, batch_size=128)


print('Train Metrics\n-------------------------')
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(train_labels, pred_train_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(train_labels, pred_train_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(train_labels, pred_train_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(train_labels, pred_train_classes)
print('F1 score: %f' % f1)
f1M = f1_score(train_labels, pred_train_classes, average='macro')
print('F1-Macro score: %f' % f1M)
# confusion matrix
matrix = confusion_matrix(train_labels, pred_train_classes)
print(matrix)

print('Test Metrics\n-------------------------')
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_labels, pred_test_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(test_labels, pred_test_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(test_labels, pred_test_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test_labels, pred_test_classes)
print('F1 score: %f' % f1)
f1M = f1_score(test_labels, pred_test_classes, average='macro')
print('F1-Macro score: %f' % f1M)
# confusion matrix
matrix = confusion_matrix(test_labels, pred_test_classes)
print(matrix)

# %%
