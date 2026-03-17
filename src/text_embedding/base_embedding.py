from abc import abstractmethod
import numpy as np

class BaseEmbedding:
    
    @abstractmethod
    def process_sentence(self,sentence: str):
        raise NotImplementedError
    
    @abstractmethod
    def get_embedding(self,word: str):
        raise NotImplementedError
    
    @abstractmethod
    def update_embedding(self, word: str, gradient: np.ndarray, lr: float):
        raise NotImplementedError
    
    @abstractmethod
    def cache_embedding(self, path:str):
        raise NotImplementedError
    
    @abstractmethod
    def load_embedding_cache(self, path:str):
        raise NotImplementedError