
# AITextClassifier

AITextClassifier is a project for detecting whether a text was written by a **human or by a large language model (LLM)** and, if generated, identifying **which model produced it**.

The current implementation uses a **TF-IDF based feature representation combined with a multiclass logistic regression classifier implemented in NumPy**.

---

## Used Datasets

### OTB
https://huggingface.co/datasets/MLNTeam-Unical/OpenTuringBench

### HC3
Human ChatGPT Comparison Corpus

### GSINGH1
https://huggingface.co/datasets/gsingh1-py/train

---

## Current Data

### Entries (`records.json`)

- chatgpt: 86279
- mistral: 52650
- gemma: 49627 (Gemini-based model)
- llama: 50800
- human: 139676

### From Sources

- HC3 samples: 197552
- GSINGH1 samples: 60161
- OTB samples: 121319

**Total samples:** 379032

---

## Dataset Scheme (`scheme.json`)

```json
{
  "model": "chatgpt,human,mistral,llama,gemma",
  "text": "afioweogwHO",
  "topic": "chemistry, physics,...",
  "length": 120,
  "origin": "hc3 or gsingh1"
}
```

| Field | Description |
|------|-------------|
| model | label indicating the generating model or human |
| text | the text sample |
| topic | topic category |
| length | text length |
| origin | dataset source |

---

## Models

### TF-IDF

The current baseline model uses **TF-IDF feature extraction combined with a multiclass logistic regression classifier** implemented in NumPy.

#### Classification Pipeline

```
text
 ↓
tokenization
 ↓
TF-IDF feature extraction
    • word n-grams (1–2)
    • character n-grams (3–5)
 ↓
feature concatenation
 ↓
softmax logistic regression
 ↓
model prediction
```

The classifier predicts one of five classes:

```
human
chatgpt
mistral
gemma
llama
```

---

#### Feature Representation

Two TF-IDF feature spaces are constructed.

##### Word n-grams

- `ngram_range = (1,2)`
- captures vocabulary usage and short phrases

Example:

```
tokens = ["this", "is", "ai"]

unigrams:
this
is
ai

bigrams:
this is
is ai
```

##### Character n-grams

- `ngram_range = (3,5)`
- captures stylistic patterns such as punctuation, whitespace usage, and tokenization artifacts

Example:

```
text = "text"

3-grams:
tex
ext

4-grams:
text
```

Character features are particularly useful for **LLM detection**, because models often have subtle stylistic signatures.

---

#### Classifier

The classifier is a **multiclass softmax logistic regression model** trained with:

- cross-entropy loss
- L2 regularization
- mini-batch gradient descent

The model learns the linear mapping:

```
logits = XW + b
```

Where:

- `X` = TF-IDF feature matrix
- `W` = weight matrix
- `b` = bias vector

Softmax converts logits into probabilities for each class.

---

#### Model Storage

After training, the model is serialized using `pickle`.

The saved model contains:

```
W           classifier weights
b           classifier bias
word_vocab  word TF-IDF vocabulary
word_idf    word IDF values
char_vocab  character TF-IDF vocabulary
char_idf    character IDF values
```

Saved models are stored in:

```
models/
```

Example filename:

```
models/tfidf_logreg_size100000_word10000_char10000.pkl
```

---

### Project Structure

```
DL-AITextClassifier
│
├── datasets/
│   └── records.json
│
├── models/
│   └── saved trained models (.pkl)
│
├── src/
│   │
│   ├── dataloader/
│   │   └── dataloader.py
│   │
│   ├── tfidf/
│   │   └── tfidf.py
│   │
│   ├── models/
│   │   └── logreg.py
│   │
│   └── train/
│       └── train.py
│
├── tests/
│   ├── test_tfidf.py
│   └── test_softmax_logreg.py
│
└── README.md
```

| Directory | Purpose |
|----------|--------|
| `datasets/` | merged dataset files |
| `models/` | saved trained models |
| `src/` | implementation code |
| `tests/` | unit tests for components |

---

### Training

Run the training pipeline with:

```
python -m src.train.train_logreg
```

This will:

1. load the dataset
2. build TF-IDF features
3. train the logistic regression classifier
4. evaluate on the test set
5. store the trained model in `models/`

---

### Baseline Performance

Typical configuration:

```
dataset size: 100000
word features: 10000
char features: 10000
epochs: 30
```

Expected accuracy:

```
≈ 0.60 – 0.70
```

This provides a strong classical baseline before experimenting with deeper models.
