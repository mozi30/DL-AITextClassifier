import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_embedding import WordEmbedding
from src.dataloader import SentenceDataModule
from src.models.gru_numpy import DenseLayer, BiGRULayer, EmbeddingLayer, SequenceMeanMaxPool, DropoutLayer, NeuralNetwork, CrossEntropyLoss, SoftmaxActivation, accuracy

if __name__ == '__main__':
    seed = 42
    record_path = str(ROOT / "datasets" / "records.json")
    
    # training data
    dm = SentenceDataModule(record_path=record_path, size=100000, split=(70, 20, 10), seed=seed)
    embedding_size = 128
    embedding = WordEmbedding(dm.get_most_common_words(top_n=30000), embedding_size=embedding_size, seed=seed)

    # network
    net = NeuralNetwork(
        dm,
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        verbose=True,
        loss=CrossEntropyLoss,
        metric=accuracy,
        early_stopping_patience=10,
        lr_patience=4,
        lr_decay_factor=0.5,
        min_learning_rate=1e-5,
    )
    
    net.add(EmbeddingLayer(embedding=embedding))
    net.add(BiGRULayer(num_input=embedding_size, num_hidden=64, return_sequences=True))
    net.add(SequenceMeanMaxPool())
    net.add(DropoutLayer(rate=0.3, seed=seed))
    net.add(DenseLayer(5))
    net.add(SoftmaxActivation())

    # train
    net.fit()

    # test
    out = net.predict()
    print(net.score(out))