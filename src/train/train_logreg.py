import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from src.dataloader.dataloader import SentenceDataModule
from src.tfidf.tfidf import TfIdfVectorizerNumpy
from src.models.logreg import SoftmaxLogReg


# Hyperparameters --------------------------------------------------------------------------------
size = 5000

word_max_features = 10000
char_max_features = 10000

min_df = 2

word_ngram_range = (1,2)
char_ngram_range = (2,6)

lr = 0.3
epochs = 40
# -----------------------------------------------------------------------------------------------

MODEL_PATH = Path(
    f"models/tfidf_logreg_size{size}_word{word_max_features}_char{char_max_features}_LONG.pkl"
)


def save_model(model, word_vectorizer, char_vectorizer, path, config, label_to_id, save_full_model=True):
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        # manual parameters (robust)
        "W": model.W,
        "b": model.b,

        # vectorizers
        "word_vocab": word_vectorizer.vocab,
        "word_idf": word_vectorizer.idf,

        "char_vocab": char_vectorizer.vocab,
        "char_idf": char_vectorizer.idf,

        # metadata
        "config": config,
        "label_to_id": label_to_id,
    }

    # optionally store full model
    if save_full_model:
        data["model"] = model

    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"Model saved to: {path}")


def main():

    print("Loading dataset...")

    dm = SentenceDataModule(
        record_path="datasets/records_long.json",
        size=size,
        split=(70, 20, 10),
    )

    train_samples = dm.get_train_loader().samples
    val_samples = dm.get_val_loader().samples
    test_samples = dm.get_test_loader().samples


    # Extract text + labels ----------------------------------------------------------------------
    X_train_text = [s["text"] for s in train_samples]
    y_train = np.array([s["model"] for s in train_samples])

    X_val_text = [s["text"] for s in val_samples]
    y_val = np.array([s["model"] for s in val_samples])

    X_test_text = [s["text"] for s in test_samples]
    y_test = np.array([s["model"] for s in test_samples])
    # --------------------------------------------------------------------------------------------


    print("Building WORD TF-IDF...")

    word_vectorizer = TfIdfVectorizerNumpy(
        max_features=word_max_features,
        min_df=min_df,
        ngram_range=word_ngram_range,
        analyzer="word",
    )

    X_train_word = word_vectorizer.fit_transform(X_train_text)
    X_val_word = word_vectorizer.transform(X_val_text)
    X_test_word = word_vectorizer.transform(X_test_text)


    print("Building CHAR TF-IDF...")

    char_vectorizer = TfIdfVectorizerNumpy(
        max_features=char_max_features,
        min_df=min_df,
        ngram_range=char_ngram_range,
        analyzer="char",
    )

    X_train_char = char_vectorizer.fit_transform(X_train_text)
    X_val_char = char_vectorizer.transform(X_val_text)
    X_test_char = char_vectorizer.transform(X_test_text)


    # Combine features ---------------------------------------------------------------------------
    X_train = np.hstack([X_train_word, X_train_char])
    X_val = np.hstack([X_val_word, X_val_char])
    X_test = np.hstack([X_test_word, X_test_char])
    print("Combined feature shape:", X_train.shape)
    # --------------------------------------------------------------------------------------------

    # ADD LENGTH FEATURE -------------------------------------------------------------------------
    length_train = np.array([len(t) for t in X_train_text]).reshape(-1, 1)
    length_val = np.array([len(t) for t in X_val_text]).reshape(-1, 1)
    length_test = np.array([len(t) for t in X_test_text]).reshape(-1, 1)

    # normalize length (so scale matches TF-IDF values)
    length_train = length_train / 200
    length_val = length_val / 200
    length_test = length_test / 200

    X_train = np.hstack([X_train, length_train])
    X_val = np.hstack([X_val, length_val])
    X_test = np.hstack([X_test, length_test])

    print("Feature shape with length:", X_train.shape)
    # -------------------------------------------------------------------------------------------

    print("Training classifier...")

    clf = SoftmaxLogReg(
        input_dim=X_train.shape[1],
        num_classes=5,
        lr=lr,
    )

    clf.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
    )


    print("Evaluating on test set...")

    preds = clf.predict(X_test)
    acc = np.mean(preds == y_test)

    # print(confusion_matrix(y_test, preds))
    # print(classification_report(y_test, preds))

    print("Test accuracy:", acc)

    save_model(clf, word_vectorizer, char_vectorizer, MODEL_PATH)

    print("Model saved to", MODEL_PATH)

    print("\nExperiment configuration:")
    print("size:", size)
    print("word_max_features:", word_max_features)
    print("char_max_features:", char_max_features)
    print("word_ngram_range:", word_ngram_range)
    print("char_ngram_range:", char_ngram_range)
    print("min_df:", min_df)
    print("lr:", lr)
    print("epochs:", epochs)


if __name__ == "__main__":
    main()