# AITextClassifier

AITextClassifier is a project for detecting whether a text was written by a **human or by a large language model (LLM)** and, if generated, identifying **which model produced it**.

The current project contains:
- a **TF-IDF + multiclass logistic regression baseline implemented in NumPy**
- a **TF-IDF + logistic regression implementation in PyTorch**
- additional experiments for **embeddings + bidirectional LSTM**

---

## Used Datasets

### OTB
https://huggingface.co/datasets/MLNTeam-Unical/OpenTuringBench

### HC3
Human ChatGPT Comparison Corpus

### GSINGH1
https://huggingface.co/datasets/gsingh1-py/train

---

## Build Datasets With One Script

Use one script to generate `datasets/records_long.json` end-to-end:

```bash
python src/dataset/build_all_records.py --min-words 100 --max-words 150
```

What this script does:

- builds source-specific intermediate files:
  - `datasets/hc3/hc3-records.json`
  - `datasets/gsingh1/gsingh1-records.json`
  - `datasets/otb/otb-records.json`
  - `datasets/anthropic/anthropic-records.json`
- writes the final long-text dataset to:
  - `datasets/records_long.json`
- builds records where word count satisfies:
  - `min_words < words < max_words`

Optional flags:

```bash
# Download OTB / GSINGH1 source files first (when available)
python src/dataset/build_all_records.py --download-sources

# Custom long-text word range
python src/dataset/build_all_records.py --min-words 100 --max-words 150
```

Download behavior:

- if `datasets/otb/all.json` already exists, OTB download is skipped
- if `datasets/gsingh1/all.json` already exists, GSINGH1 download is skipped

---

---

### Entries (`records_long.json`)

Generated with:

```bash
python src/dataset/build_all_records.py --min-words 100 --max-words 150
```

Class distribution:

- human: 55276
- chatgpt: 49981
- mistral: 79431
- gemma: 67864
- llama: 82865

### From Sources

- HC3 samples: 62149
- GSINGH1 samples: 85508
- OTB samples: 187760

**Total samples:** 335417

---

## Dataset Scheme (`scheme.json`)

```json
{
  "model": "chatgpt,human,mistral,llama,gemma",
  "text": "afioweogwHO",
  "topic": "chemistry, physics,...",
  "length": 520,
  "origin": "hc3, gsingh1 or otb"
}
```

| Field | Description |
|------|-------------|
| `model` | label indicating the generating model or human |
| `text` | the text sample |
| `topic` | topic category |
| `length` | text length |
| `origin` | dataset source |

---

## Models

### A. TF-IDF + Logistic Regression (NumPy)

This baseline uses **TF-IDF feature extraction combined with a multiclass softmax logistic regression classifier** implemented from scratch in NumPy.

#### Classification Pipeline

```text
text
 ↓
tokenization
 ↓
TF-IDF feature extraction
    • word n-grams (1–2)
    • character n-grams (2–6)
 ↓
feature concatenation
 ↓
length feature
 ↓
softmax logistic regression
 ↓
model prediction
```

The classifier predicts one of five classes:

```text
human
chatgpt
mistral
gemma
llama
```

#### Feature Representation

Two TF-IDF feature spaces are constructed.

##### Word n-grams

- `ngram_range = (1, 2)`
- captures vocabulary usage and short phrases

Example:

```text
tokens = ["this", "is", "ai"]
unigrams: "this" "is" "ai"
bigrams: "this is" "is ai"
```

##### Character n-grams

- `ngram_range = (2, 6)`
- captures stylistic patterns such as punctuation, whitespace usage, and tokenization artifacts

Example:

```text
text = "text"
2-grams: "te" "ex" "xt"
3-grams: "tex" "ext"
4-grams: "text"
```

#### Classifier

The classifier is trained with:

- cross-entropy loss
- L2 regularization
- mini-batch gradient descent

The model learns the linear mapping:

```text
logits = XW + b
```

Where:

- `X` = TF-IDF feature matrix
- `W` = weight matrix
- `b` = bias vector

Softmax converts logits into probabilities for each class.

#### Model Storage

After training, the model is serialized using `pickle`.

The saved model contains:

```text
W           classifier weights
b           classifier bias
word_vocab  word TF-IDF vocabulary
word_idf    word IDF values
char_vocab  character TF-IDF vocabulary
char_idf    character IDF values
```

Saved models are stored in:

```text
models/
```

Example filename:

```text
models/subm1-g5-MEI-A.pkl
```

---

### B. TF-IDF + Logistic Regression (PyTorch)

A second submission model uses the same **TF-IDF feature representation**, but the classifier is implemented in **PyTorch** instead of NumPy.

This model is used for the `subm1-g5-MEI-B` submission.

---

## Baseline Performance

Current best result for the NumPy TF-IDF + logistic regression model:

```text
size: 100000
word_max_features: 15000
char_max_features: 15000
min_df: 2
word_ngram_range: (1, 2)
char_ngram_range: (2, 6)
lr: 0.3
epochs: 40
```

Test accuracy:

```text
≈ 0.81
```

---

### C. DNN with MultiHeadAttention

This model uses ether Word Encoding or NgramEncoding for text representation combined with a deep neuronal network approach thats making use of Multi Head Attention to get better understanding of text structure and dependencies. 

The model implementation can be found under:

```bash
src/models/gru_numpy
```

To run the model with pretrained weights use the command
```python
 net.load_weights(weights_path)
```

A test implementation can be found in 

```bash
tests/test_numpy_mha.py
```

Test accuracy:

```text
0.787
```

## Project Structure

```text
DL-AITextClassifier
│
├── cache/
│   ├── numpy_mha_trained.pkl
│
├── datasets/
│   ├── records_long.json
│
├── models/
│   └── subm1-g5-MEI-A.pkl
│
├── notebooks/
│   ├── logReg_tfidf.ipynb
│
├── src/
│   ├── dataloader/
│   │   ├── __init__.py
│   │   └── dataloader.py
│   │
│   ├── dataset/
│   │   ├── build_all_records.py
│   │
│   ├── models/
│   │   ├── gru_numpy/
│   │   ├── __init__.py
│   │   ├── gru_pytorch.py
│   │   └── logreg.py
│   │
│   ├── text_embedding/
│   │   ├── __init__.py
│   │   ├── base_embedding.py
│   │   ├── ngram_embedding.py
│   │   └── word_embedding.py
│   │
│   ├── tfidf/
│   │   ├── __init__.py
│   │   └── tfidf.py
│   │
│   └── train/
│       ├── __init__.py
│       └── train_logreg.py
│
├── Subm1/
│   ├── subm1-g5-MEI-A.csv
│   ├── subm1-g5-MEI-A.ipynb
│   ├── subm1-g5-MEI-B.csv
│   └── subm1-g5-MEI-B.ipynb
│
├── tests/
│   ├── TestData/
│   ├── __init__.py
│   ├── test_numpy_gru.py
│   ├── test_softmax_logreg.py
│   └── test_tfidf.py
│
├── .gitignore
├── __init__.py
├── INSTRUCTIONS_EN.md
├── README.md
├── requirements.txt
└── results.txt
```
