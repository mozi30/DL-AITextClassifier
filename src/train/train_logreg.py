import numpy as np
import pickle
from pathlib import Path

from src.dataloader.dataloader import SentenceDataModule
from src.tfidf.tfidf import TfIdfVectorizerNumpy
from src.models.logreg import SoftmaxLogReg


# Hyperparameters --------------------------------------------------------------------------------
size = 100000

word_max_features = 10000
char_max_features = 10000

min_df = 2

word_ngram_range = (1,2)
char_ngram_range = (3,5)

lr = 0.2
epochs = 30
# -----------------------------------------------------------------------------------------------

MODEL_PATH = Path(
    f"models/tfidf_logreg_size{size}_word{word_max_features}_char{char_max_features}.pkl"
)


def save_model(model, word_vectorizer, char_vectorizer, path):
    path.parent.mkdir(exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(
            {
                "W": model.W,
                "b": model.b,

                "word_vocab": word_vectorizer.vocab,
                "word_idf": word_vectorizer.idf,

                "char_vocab": char_vectorizer.vocab,
                "char_idf": char_vectorizer.idf,
            },
            f,
        )


def main():

    print("Loading dataset...")

    dm = SentenceDataModule(
        record_path="datasets/records.json",
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

    print("Test accuracy:", acc)


    print("Saving model...")

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