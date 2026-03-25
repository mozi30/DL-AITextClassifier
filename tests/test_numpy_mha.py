import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_embedding import NgramEmbedding
from src.dataloader import SentenceDataModule
from src.models.gru_numpy import DenseLayer, MultiHeadAttentionLayer, EmbeddingLayer, SequenceMeanMaxPool, DropoutLayer, NeuralNetwork, CrossEntropyLoss, SoftmaxActivation, accuracy 

if __name__ == '__main__':
    seed = 42
    record_path = str(ROOT / "datasets" / "records_long.json")
    
    # -------------- Insert mha weights path here -----------------------
    weights_path = str(ROOT / "cache" / "numpy_mha_trained.pkl")
    current_epoch_weights_path = str(ROOT / "cache" / "numpy_mha_current_epoch.pkl")
    best_epoch_weights_path = str(ROOT / "cache" / "numpy_mha_best_epoch.pkl")
    
    
    # training data
    dm = SentenceDataModule(record_path=record_path, size=100000, split=(70, 20, 10), seed=seed)
    embedding_size = 128
    #embedding = WordEmbedding(dm.get_most_common_words(top_n=30000), embedding_size=embedding_size, seed=seed)
    embedding = NgramEmbedding(dm.get_most_common_words(top_n=30000),embedding_size,seed=seed)

    # network
    net = NeuralNetwork(
        dm,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        verbose=True,
        loss=CrossEntropyLoss,
        metric=accuracy,
        early_stopping_patience=10,
        lr_patience=4,
        lr_decay_factor=0.5,
        min_learning_rate=1e-5,
        num_train_epoch_batches=100,
        num_val_epoch_batches=10,
        save_current_epoch_weights_path=current_epoch_weights_path,
        save_best_epoch_weights_path=best_epoch_weights_path,
    )
    
    net.add(EmbeddingLayer(embedding=embedding))
    net.add(MultiHeadAttentionLayer(num_heads=4,model_dim=embedding_size, seed=seed))
    net.add(SequenceMeanMaxPool())
    net.add(DropoutLayer(rate=0.3, seed=seed))
    net.add(DenseLayer(5))
    net.add(SoftmaxActivation())

    net.load_weights(weights_path)
    #train
    net.fit()
    net.save_weights(weights_path)

    # test
    out = net.predict()
    print(net.score(out))
   
    
    