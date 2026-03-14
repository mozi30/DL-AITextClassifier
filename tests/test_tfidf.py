import numpy as np

from src.dataloader.dataloader import SentenceDataModule
from src.tfidf.tfidf import TfIdfVectorizerNumpy
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


# WORD N-GRAM TF-IDF ---------------------------------------------------------------------------------------------------
vectorizer_word = TfIdfVectorizerNumpy(
    max_features=3000,
    min_df=2,
    ngram_range=(1,2),
    analyzer="word"
)

print("Building WORD TF-IDF...")

X_train_word = vectorizer_word.fit_transform(X_train_text)
X_val_word = vectorizer_word.transform(X_val_text)
X_test_word = vectorizer_word.transform(X_test_text)

print("Word Train shape:", X_train_word.shape)
print("Word Val shape:", X_val_word.shape)
print("Word Test shape:", X_test_word.shape)
# ----------------------------------------------------------------------------------------------------------------------

# CHARACTER N-GRAM TF-IDF ----------------------------------------------------------------------------------------------
vectorizer_char = TfIdfVectorizerNumpy(
    max_features=3000,
    min_df=2,
    ngram_range=(3,5),
    analyzer="char"
)

print("Building CHAR TF-IDF...")

X_train_char = vectorizer_char.fit_transform(X_train_text)
X_val_char = vectorizer_char.transform(X_val_text)
X_test_char = vectorizer_char.transform(X_test_text)

print("Char Train shape:", X_train_char.shape)
print("Char Val shape:", X_val_char.shape)
print("Char Test shape:", X_test_char.shape)
# ----------------------------------------------------------------------------------------------------------------------