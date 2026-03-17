
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

### Entries (`records_long.json`)

- mistral: 80,999
- chatgpt: 35,167
- gemma: 69,056
- llama: 85,616
- human: 49,517

### From Sources

- HC3 samples: 84,684 
- OTB samples: 235,671

**Total samples:** 320,355


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
    • character n-grams (2–6)
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
unigrams: "this" "is" "ai"
bigrams: "this is" "is ai"
```

##### Character n-grams

- `ngram_range = (3,5)`
- captures stylistic patterns such as punctuation, whitespace usage, and tokenization artifacts

Example:

```
text = "text"
3-grams: "tex" "ext"
4-grams: "text"
```

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

---

### Training

Run the training pipeline with and set the hyperparameters in this file:

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

## Build Datasets With One Script

Use one script to generate `datasets/records_long.json` end-to-end:

```bash
python src/dataset/build_all_records.py
```

What this script does:

- builds source-specific intermediate files:
  - `datasets/hc3/hc3-records.json`
  - `datasets/gsingh1/gsingh1-records.json` (only if source exists)
  - `datasets/otb/otb-records.json`
- writes final long-text dataset to:
  - `datasets/records_long.json`
- keeps only records where word count satisfies:
  - `min_words < words < max_words` (defaults: 80 and 200)

Optional flags:

```bash
# Download OTB/GSINGH1 source files first (when available)
python src/dataset/build_all_records.py --download-sources

# Custom long-text word range
python src/dataset/build_all_records.py --min-words 100 --max-words 220
```

Download behavior:

- if `datasets/otb/all.json` already exists, OTB download is skipped
- if `datasets/gsingh1/all.json` already exists, GSINGH1 download is skipped

---

### Baseline Performance

Current best result with this configuration:

```
dataset size: 100000
word features: 15000
char features: 15000
epochs: 40
word_ngram_range = (1,2)
char_ngram_range = (2,6)
lr = 0.3
```

Test accuracy:

```
≈ 0.61
```

