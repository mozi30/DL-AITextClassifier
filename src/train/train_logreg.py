import numpy as np
import pickle
from pathlib import Path

from src.dataloader.dataloader import SentenceDataModule
from src.tfidf.tfidf import TfIdfVectorizerNumpy
from src.models.logreg import SoftmaxLogReg


# TODO set vals --------------------------------------------------------------------------------------------------------
size = 100000
max_features = 20000
min_df = 2
ngram_range = (1,2)
lr = 0.2
epochs = 30
# ----------------------------------------------------------------------------------------------------------------------

MODEL_PATH = Path("models/tfidf_logreg_size" + str(size) + "_features" + str(max_features) + ".pkl")


def save_model(model, vectorizer, path):
    path.parent.mkdir(exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(
            {
                "W": model.W,
                "b": model.b,
                "vocab": vectorizer.vocab,
                "idf": vectorizer.idf,
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

    X_train_text = [s["text"] for s in train_samples]
    y_train = np.array([s["model"] for s in train_samples])

    X_val_text = [s["text"] for s in val_samples]
    y_val = np.array([s["model"] for s in val_samples])

    X_test_text = [s["text"] for s in test_samples]
    y_test = np.array([s["model"] for s in test_samples])

    print("Building TF-IDF features...")

    vectorizer = TfIdfVectorizerNumpy(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

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

    save_model(clf, vectorizer, MODEL_PATH)

    print("Model saved to", MODEL_PATH)

    print("size: ", size)
    print("max_features: ", max_features)
    print("ngram_range: ", ngram_range)
    print("mind_df: ", min_df)
    print("lr: ", lr)
    print("epochs: ", epochs)


if __name__ == "__main__":
    main()