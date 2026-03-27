import pickle

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