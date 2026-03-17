import os
import re
import numpy as np

from .base_embedding import BaseEmbedding


class NgramEmbedding(BaseEmbedding):
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        vocabulary: list[str],
        embedding_size: int = 300,
        seed: int = 42,
        use_words: bool = True,
        use_word_ngrams: bool = True,
        use_char_ngrams: bool = True,
        word_ngram_range: tuple[int, int] = (2, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        cache: bool = True,
        cache_path: str | None = None,
        cache_save_interval: int = 1000000,
    ):
        self.pad_token = self.PAD_TOKEN
        self.unk_token = self.UNK_TOKEN

        self.embedding_size = embedding_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.use_words = use_words
        self.use_word_ngrams = use_word_ngrams
        self.use_char_ngrams = use_char_ngrams

        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range

        self.vocabulary = self._with_special_tokens(self._normalize_vocabulary(vocabulary))

        self.cache = cache
        self.cache_path = cache_path or "./cache/ngram_embedding.npz"
        self.cache_save_interval = cache_save_interval
        self.embedding_update_cnt = 0

        if self.cache and os.path.exists(self.cache_path):
            if self._is_cache_compatible(self.cache_path):
                self.load_embedding_cache(self.cache_path)
            else:
                self._initialize_embeddings()
                self.cache_embedding()
        else:
            self._initialize_embeddings()
            if self.cache:
                self.cache_embedding()

    @staticmethod
    def _normalize_vocabulary(vocabulary):
        normalized = []
        for item in vocabulary:
            if isinstance(item, tuple):
                normalized.append(item[0])
            else:
                normalized.append(item)

        seen = set()
        unique = []
        for token in normalized:
            if token not in seen:
                unique.append(token)
                seen.add(token)
        return unique

    def _with_special_tokens(self, vocabulary):
        filtered = [token for token in vocabulary if token not in {self.pad_token, self.unk_token}]
        return [self.pad_token, self.unk_token, *filtered]

    def _build_lookup(self, items):
        item_to_id = {}
        id_to_item = {}
        for idx, item in enumerate(items):
            item_to_id[item] = idx
            id_to_item[idx] = item
        return item_to_id, id_to_item

    def _init_embedding_matrix(self, size):
        if size == 0:
            return np.empty((0, self.embedding_size), dtype=float)

        matrix = np.zeros((size, self.embedding_size), dtype=float)
        for idx in range(size):
            if idx == 0 and size > 0:
                # only used for PAD in word vocab if aligned there;
                # harmless for ngram tables too
                matrix[idx] = self.rng.uniform(-0.01, 0.01, self.embedding_size)
            else:
                matrix[idx] = self.rng.uniform(-0.01, 0.01, self.embedding_size)
        return matrix

    def _initialize_embeddings(self):
        self.word_to_id, self.id_to_word = self._build_lookup(self.vocabulary)
        self.word_embeddings = self._init_embedding_matrix(len(self.vocabulary))

        pad_idx = self.word_to_id[self.pad_token]
        self.word_embeddings[pad_idx] = np.zeros(self.embedding_size, dtype=float)

        self.encoded = self.word_embeddings

        self.word_ngram_vocab = (
            self._build_word_ngram_vocab(self.vocabulary) if self.use_word_ngrams else []
        )
        self.char_ngram_vocab = (
            self._build_char_ngram_vocab(self.vocabulary) if self.use_char_ngrams else []
        )

        self.word_ngram_to_id, self.id_to_word_ngram = self._build_lookup(self.word_ngram_vocab)
        self.char_ngram_to_id, self.id_to_char_ngram = self._build_lookup(self.char_ngram_vocab)

        self.word_ngram_embeddings = (
            self._init_embedding_matrix(len(self.word_ngram_vocab))
            if self.use_word_ngrams
            else None
        )
        self.char_ngram_embeddings = (
            self._init_embedding_matrix(len(self.char_ngram_vocab))
            if self.use_char_ngrams
            else None
        )

    def _is_cache_compatible(self, path: str) -> bool:
        try:
            data = np.load(path, allow_pickle=True)
            cached_vocabulary = data["vocabulary"].tolist()
            cached_embedding_size = int(data["embedding_size"])
            cached_use_words = bool(data["use_words"])
            cached_use_word_ngrams = bool(data["use_word_ngrams"])
            cached_use_char_ngrams = bool(data["use_char_ngrams"])
            cached_word_ngram_range = tuple(data["word_ngram_range"].tolist())
            cached_char_ngram_range = tuple(data["char_ngram_range"].tolist())

            return (
                cached_vocabulary == self.vocabulary
                and cached_embedding_size == self.embedding_size
                and cached_use_words == self.use_words
                and cached_use_word_ngrams == self.use_word_ngrams
                and cached_use_char_ngrams == self.use_char_ngrams
                and cached_word_ngram_range == self.word_ngram_range
                and cached_char_ngram_range == self.char_ngram_range
            )
        except Exception:
            return False

    @staticmethod
    def _split_into_words(sentence: str):
        return re.findall(r"\b\w+\b", sentence.lower())

    def _lookup_word(self, word: str):
        return word if word in self.word_to_id else self.unk_token

    def _word_ngrams_from_tokens(self, tokens: list[str]):
        min_n, max_n = self.word_ngram_range
        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append("_".join(tokens[i:i + n]))
        return ngrams

    def _char_ngrams_from_word(self, word: str):
        min_n, max_n = self.char_ngram_range
        word = f"<{word}>"
        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i + n])
        return ngrams

    def _build_word_ngram_vocab(self, vocabulary: list[str]):
        vocab = set()
        real_words = [w for w in vocabulary if w not in {self.pad_token, self.unk_token}]
        for word in real_words:
            # no sentence-level word ngrams can be built from isolated words alone
            _ = word
        return sorted(vocab)

    def _build_char_ngram_vocab(self, vocabulary: list[str]):
        vocab = set()
        real_words = [w for w in vocabulary if w not in {self.pad_token, self.unk_token}]
        for word in real_words:
            vocab.update(self._char_ngrams_from_word(word))
        return sorted(vocab)

    def build_word_ngram_vocab_from_corpus(self, sentences: list[str]):
        if not self.use_word_ngrams:
            return

        vocab = set()
        for sentence in sentences:
            tokens = [self._lookup_word(w) for w in self._split_into_words(sentence)]
            vocab.update(self._word_ngrams_from_tokens(tokens))

        self.word_ngram_vocab = sorted(vocab)
        self.word_ngram_to_id, self.id_to_word_ngram = self._build_lookup(self.word_ngram_vocab)
        self.word_ngram_embeddings = self._init_embedding_matrix(len(self.word_ngram_vocab))

        if self.cache:
            self.cache_embedding()

    def get_word_embedding(self, word: str):
        token = self._lookup_word(word)
        idx = self.word_to_id[token]
        return self.word_embeddings[idx]

    def get_char_ngram_embedding(self, ngram: str):
        if not self.use_char_ngrams or ngram not in self.char_ngram_to_id:
            return None
        idx = self.char_ngram_to_id[ngram]
        return self.char_ngram_embeddings[idx]

    def get_word_ngram_embedding(self, ngram: str):
        if not self.use_word_ngrams or ngram not in self.word_ngram_to_id:
            return None
        idx = self.word_ngram_to_id[ngram]
        return self.word_ngram_embeddings[idx]

    def get_combined_word_embedding(self, word: str):
        vectors = []

        if self.use_words:
            vectors.append(self.get_word_embedding(word))

        if self.use_char_ngrams:
            char_ngrams = self._char_ngrams_from_word(word)
            char_vectors = []
            for ng in char_ngrams:
                vec = self.get_char_ngram_embedding(ng)
                if vec is not None:
                    char_vectors.append(vec)
            if char_vectors:
                vectors.append(np.mean(char_vectors, axis=0))

        if not vectors:
            return np.zeros(self.embedding_size, dtype=float)

        return np.mean(vectors, axis=0)

    def process_sentence(self, sentence: str):
        raw_words = self._split_into_words(sentence)
        tokens = [self._lookup_word(w) for w in raw_words]

        embeddings = []

        for i, word in enumerate(raw_words):
            vectors = []

            if self.use_words:
                vectors.append(self.get_word_embedding(word))

            if self.use_char_ngrams:
                char_ngrams = self._char_ngrams_from_word(word)
                char_vectors = []

                for ng in char_ngrams:
                    vec = self.get_char_ngram_embedding(ng)
                    if vec is not None:
                        char_vectors.append(vec)

                if char_vectors:
                    vectors.append(np.mean(char_vectors, axis=0))

            if self.use_word_ngrams:
                min_n, max_n = self.word_ngram_range
                context_vectors = []

                for n in range(min_n, max_n + 1):
                    start = max(0, i - n + 1)
                    end = min(len(tokens) - n + 1, i + 1)

                    for j in range(start, end):
                        ng = "_".join(tokens[j:j + n])
                        vec = self.get_word_ngram_embedding(ng)
                        if vec is not None:
                            context_vectors.append(vec)

                if context_vectors:
                    vectors.append(np.mean(context_vectors, axis=0))

            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.embedding_size, dtype=float))

        if embeddings:
            embeddings = np.asarray(embeddings, dtype=float)
        else:
            embeddings = np.empty((0, self.embedding_size), dtype=float)

        return tokens, embeddings

    def get_sentence_embedding(self, sentence: str, include_word_ngrams: bool = True):
        _, token_embeddings = self.process_sentence(sentence)
        vectors = []

        if len(token_embeddings) > 0:
            vectors.append(np.mean(token_embeddings, axis=0))

        if include_word_ngrams and self.use_word_ngrams:
            tokens = [self._lookup_word(w) for w in self._split_into_words(sentence)]
            sentence_ngrams = self._word_ngrams_from_tokens(tokens)
            ng_vectors = []

            for ng in sentence_ngrams:
                vec = self.get_word_ngram_embedding(ng)
                if vec is not None:
                    ng_vectors.append(vec)

            if ng_vectors:
                vectors.append(np.mean(ng_vectors, axis=0))

        if not vectors:
            return np.zeros(self.embedding_size, dtype=float)

        return np.mean(vectors, axis=0)

    def get_embedding(self, word: str):
        return self.get_combined_word_embedding(word)

    def update_embedding(self, word: str, gradient: np.ndarray, lr: float = 0.01):
        token = self._lookup_word(word)
        if token == self.pad_token:
            return

        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != (self.embedding_size,):
            raise ValueError(
                f"gradient shape must be ({self.embedding_size},), got {gradient.shape}"
            )

        self.update_word_embedding(token, gradient, lr=lr)
        self.update_char_ngram_embeddings(token, gradient, lr=lr)

        self.embedding_update_cnt += 1
        if self.cache and self.embedding_update_cnt >= self.cache_save_interval:
            self.embedding_update_cnt = 0
            self.cache_embedding()

    def update_word_embedding(self, word: str, gradient: np.ndarray, lr: float = 0.01):
        token = self._lookup_word(word)
        if token == self.pad_token:
            return
        idx = self.word_to_id[token]
        self.word_embeddings[idx] -= lr * gradient
        

    def update_char_ngram_embeddings(self, word: str, gradient: np.ndarray, lr: float = 0.01):
        if not self.use_char_ngrams:
            return

        ngrams = self._char_ngrams_from_word(word)
        valid_ids = [self.char_ngram_to_id[ng] for ng in ngrams if ng in self.char_ngram_to_id]
        if not valid_ids:
            return

        shared_grad = gradient / len(valid_ids)
        for idx in valid_ids:
            self.char_ngram_embeddings[idx] -= lr * shared_grad

    def update_word_ngram_embedding(self, ngram: str, gradient: np.ndarray, lr: float = 0.01):
        if not self.use_word_ngrams or ngram not in self.word_ngram_to_id:
            return
        idx = self.word_ngram_to_id[ngram]
        self.word_ngram_embeddings[idx] -= lr * gradient

        self.embedding_update_cnt += 1
        if self.cache and self.embedding_update_cnt >= self.cache_save_interval:
            self.embedding_update_cnt = 0
            self.cache_embedding()

    def cache_embedding(self, path: str | None = None) -> None:
        path = path or self.cache_path

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        print("Cached")
        np.savez_compressed(
            path,
            vocabulary=np.array(self.vocabulary, dtype=object),
            embedding_size=self.embedding_size,
            seed=self.seed,
            use_words=self.use_words,
            use_word_ngrams=self.use_word_ngrams,
            use_char_ngrams=self.use_char_ngrams,
            word_ngram_range=np.array(self.word_ngram_range, dtype=int),
            char_ngram_range=np.array(self.char_ngram_range, dtype=int),
            word_embeddings=self.word_embeddings,
            word_ngram_vocab=np.array(self.word_ngram_vocab, dtype=object),
            char_ngram_vocab=np.array(self.char_ngram_vocab, dtype=object),
            word_ngram_embeddings=(
                self.word_ngram_embeddings
                if self.word_ngram_embeddings is not None
                else np.empty((0, self.embedding_size), dtype=float)
            ),
            char_ngram_embeddings=(
                self.char_ngram_embeddings
                if self.char_ngram_embeddings is not None
                else np.empty((0, self.embedding_size), dtype=float)
            ),
        )

    def load_embedding_cache(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)

        self.vocabulary = data["vocabulary"].tolist()
        self.embedding_size = int(data["embedding_size"])
        self.seed = int(data["seed"])

        self.use_words = bool(data["use_words"])
        self.use_word_ngrams = bool(data["use_word_ngrams"])
        self.use_char_ngrams = bool(data["use_char_ngrams"])

        self.word_ngram_range = tuple(data["word_ngram_range"].tolist())
        self.char_ngram_range = tuple(data["char_ngram_range"].tolist())

        self.rng = np.random.default_rng(self.seed)

        self.word_to_id, self.id_to_word = self._build_lookup(self.vocabulary)
        self.word_embeddings = data["word_embeddings"]
        self.encoded = self.word_embeddings

        self.word_ngram_vocab = data["word_ngram_vocab"].tolist()
        self.char_ngram_vocab = data["char_ngram_vocab"].tolist()

        self.word_ngram_to_id, self.id_to_word_ngram = self._build_lookup(self.word_ngram_vocab)
        self.char_ngram_to_id, self.id_to_char_ngram = self._build_lookup(self.char_ngram_vocab)

        loaded_word_ngram_embeddings = data["word_ngram_embeddings"]
        loaded_char_ngram_embeddings = data["char_ngram_embeddings"]

        self.word_ngram_embeddings = (
            loaded_word_ngram_embeddings if self.use_word_ngrams else None
        )
        self.char_ngram_embeddings = (
            loaded_char_ngram_embeddings if self.use_char_ngrams else None
        )

        self.pad_token = self.PAD_TOKEN
        self.unk_token = self.UNK_TOKEN