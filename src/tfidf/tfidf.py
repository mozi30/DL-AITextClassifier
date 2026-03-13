import re
import math
import numpy as np
from collections import Counter, defaultdict



class TfIdfVectorizerNumpy:
    # max_features: top 5000 unigrams and bigrams go into vocabulary (top means the terms with highest global term frequency)
    # min_df: min document frequency
    # ngram_range: only unigrams and bigrams. Example, for:
    #   tokens = ["this", "is", "good"]
    #       unigrams: ["this", "is", "good"]
    #       bigrams: ["this is", "is good"]
    def __init__(self, max_features=5000, min_df=2, ngram_range=(1, 2), lowercase=True, analyzer="word"):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.analyzer = analyzer

        self.vocab = {}
        self.idf = None

    # From: "This is AI."
    # To: ["this", "is", "ai"]
    def _tokenize(self, text: str):
        if self.lowercase:
            text = text.lower()

        # simple word tokenizer
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    # Generate n-gram features from the input text.
    # Behavior depends on the "analyzer" setting:
    #
    # analyzer = "word"
    #   Generates word-based n-grams from the token list.
    #   Example:
    #       tokens = ["this", "is", "ai"]
    #       ngram_range = (1,2)
    #   Output:
    #       ["this", "is", "ai", "this is", "is ai"]
    #
    # analyzer = "char"
    #   Generates character-based n-grams directly from the raw text string.
    #   Example:
    #       text = "text"
    #       ngram_range = (3,4)
    #   Output:
    #       ["tex", "ext", "text"]
    def _generate_ngrams(self, text, tokens):

        feats = []
        min_n, max_n = self.ngram_range

        # WORD NGRAMS
        if self.analyzer == "word":
            for n in range(min_n, max_n + 1):
                if n == 1:
                    feats.extend(tokens)
                else:
                    feats.extend(
                        [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
                    )

        # CHARACTER NGRAMS
        elif self.analyzer == "char":
            text = text.lower()

            for n in range(min_n, max_n + 1):
                feats.extend(
                    [text[i:i + n] for i in range(len(text) - n + 1)]
                )

        return feats

    def fit(self, texts):
        doc_freq = Counter()
        term_freq_global = Counter()
        N = len(texts)

        # collect document frequency and global term frequency
        for text in texts:
            tokens = self._tokenize(text)
            feats = self._generate_ngrams(text, tokens)

            unique_terms = set(feats) # each termn counts only once per document
            doc_freq.update(unique_terms)
            term_freq_global.update(feats)

        # filter by min_df
        candidates = [term for term, df in doc_freq.items() if df >= self.min_df]

        # sort by global term frequency, keep top max_features
        candidates.sort(key=lambda t: term_freq_global[t], reverse=True)
        candidates = candidates[:self.max_features]

        self.vocab = {term: idx for idx, term in enumerate(candidates)}

        # compute idf
        D = len(self.vocab)
        self.idf = np.zeros(D, dtype=np.float32)

        for term, idx in self.vocab.items():
            df = doc_freq[term]
            # idf formula
            #    a term that appears in many documents gets lower IDF
            #    a term that appears in fewer documents gets higher IDF
            self.idf[idx] = math.log((1 + N) / (1 + df)) + 1.0

        return self

    # creates TF-IDF vectors
    def transform(self, texts):
        # empty matrix
        # N = number of documents
        # D = vocabulary size
        # X[i, j] = value of feature j in document i
        N = len(texts)
        D = len(self.vocab)
        X = np.zeros((N, D), dtype=np.float32)

        # fill term frequencies
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            feats = self._generate_ngrams(text, tokens)
            counts = Counter(feats)

            # X[i, idx] contains raw term counts
            for term, count in counts.items():
                idx = self.vocab.get(term)
                if idx is not None:
                    X[i, idx] = count

        # tfidf
        #   each term count is weighted by the term’s IDF
        #       count of "ai" in doc = 3
        #       idf of "ai" = 1.5
        #           TF-IDF value becomes 3 * 1.5 = 4.5
        X *= self.idf

        # row-wise l2 normalization
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (norms + 1e-8)

        return X

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)