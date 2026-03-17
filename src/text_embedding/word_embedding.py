import re
import numpy as np
import os
from .base_embedding import BaseEmbedding


class WordEmbedding(BaseEmbedding):
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, vocabulary: list[str], embedding_size: int = 300, seed: int = 42, cache: bool = True,  cache_path: str | None = None, cache_save_interval: int = 1000000):
        self.pad_token = self.PAD_TOKEN
        self.unk_token = self.UNK_TOKEN
        self.vocabulary = self._with_special_tokens(self._normalize_vocabulary(vocabulary))
        self.embedding_size = embedding_size
        self.vocab_size = len(self.vocabulary)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cache_save_interval = cache_save_interval
        if(cache):
            if(not cache_path):
                cache_path = "./cache/word_embedding.npz"
            self.load_embedding_cache(cache_path)
            self.cache_path = cache_path
        else:
            self.encoded, self.word_to_encoded, self.encoded_to_word = self._create_base_encoding()
        self.embedding_update_cnt = 0
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

    def _create_base_encoding(self):
        encodings = np.zeros((self.vocab_size, self.embedding_size))
        word_to_encoded = {}
        encoded_to_word = {}

        for index, word in enumerate(self.vocabulary):
            if word == self.pad_token:
                vector = np.zeros(self.embedding_size, dtype=float)
            else:
                vector = self.rng.uniform(-0.01, 0.01, self.embedding_size)
            encodings[index] = vector
            word_to_encoded[word] = index
            encoded_to_word[index] = word

        return encodings, word_to_encoded, encoded_to_word
    
    @staticmethod
    def _split_into_words(sentence: str):
        return re.findall(r"\b\w+\b", sentence.lower())
    
    def process_sentence(self,sentence: str):
        tokens = []
        embeddings = []
        words = self._split_into_words(sentence)
        for word in words:
            token = self._lookup_token(word)
            embedding = self.get_embedding(token)
            tokens.append(token)
            embeddings.append(embedding)
        if embeddings:
            embeddings = np.asarray(embeddings, dtype=float)
        else:
            embeddings = np.empty((0, self.embedding_size), dtype=float)
        return tokens, embeddings
            
    def _lookup_token(self, word: str):
        return word if word in self.word_to_encoded else self.unk_token

    
    def get_embedding(self, word: str):
        index = self.word_to_encoded.get(word, self.word_to_encoded[self.unk_token])
        return self.encoded[index]
    
    def update_embedding(self, word: str, gradient: np.ndarray, lr: float = 0.01):
        token = self._lookup_token(word)
        if token == self.pad_token:
            return
        index = self.word_to_encoded[token]
        self.encoded[index] -= lr * gradient
        self.embedding_update_cnt = self.embedding_update_cnt + 1
        if self.embedding_update_cnt >= self.cache_save_interval:
            self.embedding_update_cnt = 0
            if(self.cache):
                self.cache_embedding()
        
    def cache_embedding(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        print("Caching")
        np.savez_compressed(
            path,
            vocabulary=np.array(self.vocabulary, dtype=object),
            encoded=self.encoded,
            embedding_size=self.embedding_size,
            seed=self.seed,
        )


    def load_embedding_cache(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)

        self.vocabulary = data["vocabulary"].tolist()
        self.encoded = data["encoded"]
        self.embedding_size = int(data["embedding_size"])
        self.vocab_size = len(self.vocabulary)
        self.seed = int(data["seed"])

        self.rng = np.random.default_rng(self.seed)

        self.word_to_encoded = {
            word: idx for idx, word in enumerate(self.vocabulary)
        }
        self.encoded_to_word = {
            idx: word for idx, word in enumerate(self.vocabulary)
        }

        self.pad_token = self.PAD_TOKEN
        self.unk_token = self.UNK_TOKEN
        