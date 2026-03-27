import numpy as np

from src.dataloader.dataloader import SentenceDataModule
from src.text_embedding.tfidf import TfIdfVectorizerNumpy
from src.models.logreg import SoftmaxLogReg

# load data ------------------------------------------------------------------------------------------------------------
dm = SentenceDataModule(
    record_path="datasets/records.json",
    size=100000,
    split=(70,20,10),
    seed=42
)
train_samples = dm.get_train_loader().samples
val_samples = dm.get_val_loader().samples
test_samples = dm.get_test_loader().samples

print(train_samples[0])
print(val_samples[0])
print(test_samples[0])
# ----------------------------------------------------------------------------------------------------------------------


# Extract text and label (=model) --------------------------------------------------------------------------------------
X_train_text = [s["text"] for s in train_samples]
y_train = np.array([s["model"] for s in train_samples])

X_val_text = [s["text"] for s in val_samples]
y_val = np.array([s["model"] for s in val_samples])

X_test_text = [s["text"] for s in test_samples]
y_test = np.array([s["model"] for s in test_samples])

print(X_test_text[0])
print(y_test)
# ----------------------------------------------------------------------------------------------------------------------


# Run TF_IDF -----------------------------------------------------------------------------------------------------------
vectorizer = TfIdfVectorizerNumpy(
    max_features=10000,
    min_df=2,
    ngram_range=(1,2)
)

print("Building TF-IDF...")

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)

# shape: (rows, columns)
#   rows = documents
#   columns = vocabulary features
print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)
# ----------------------------------------------------------------------------------------------------------------------


# Compute Softmax Logic Regression -------------------------------------------------------------------------------------
clf = SoftmaxLogReg(
    input_dim=X_train.shape[1],
    num_classes=5,
    lr=0.2,
    reg=1e-4
)

clf.fit(
    X_train,
    y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=30,
    batch_size=256
)
# ----------------------------------------------------------------------------------------------------------------------