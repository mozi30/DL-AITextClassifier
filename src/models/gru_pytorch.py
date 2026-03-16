import importlib.util
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.dataloader.dataloader import SentenceDataModule


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def split_into_words(sentence: str):
	return re.findall(r"\b\w+\b", sentence.lower())


def build_vocabulary(samples, min_freq=2, max_vocab_size=20000):
	word_counter = Counter()

	for sample in samples:
		word_counter.update(split_into_words(sample["text"]))

	most_common_words = [
		word for word, count in word_counter.most_common() if count >= min_freq
	]

	if max_vocab_size is not None:
		most_common_words = most_common_words[: max(0, max_vocab_size - 2)]

	vocabulary = [PAD_TOKEN, UNK_TOKEN] + most_common_words
	word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
	return vocabulary, word_to_idx


def encode_text(text: str, word_to_idx):
	unk_idx = word_to_idx[UNK_TOKEN]
	sequence = [word_to_idx.get(word, unk_idx) for word in split_into_words(text)]
	return sequence or [unk_idx]


def _load_word_embedding_class():
	module_path = Path(__file__).resolve().parents[1] / "text-embedding" / "word_embedding.py"

	if not module_path.exists():
		return None

	spec = importlib.util.spec_from_file_location("word_embedding_module", module_path)
	if spec is None or spec.loader is None:
		return None

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return getattr(module, "WordEmbedding", None)


def build_embedding_matrix(vocabulary, embedding_dim=300, seed=42):
	word_embedding_cls = _load_word_embedding_class()

	if word_embedding_cls is None:
		rng = np.random.default_rng(seed)
		embedding_matrix = rng.uniform(-0.01, 0.01, (len(vocabulary), embedding_dim))
	else:
		embedding = word_embedding_cls(vocabulary=vocabulary, embedding_size=embedding_dim, seed=seed)
		embedding_matrix = embedding.encoded

	embedding_matrix = np.asarray(embedding_matrix, dtype=np.float32)
	embedding_matrix[0] = 0.0
	return embedding_matrix


class TextDataset(Dataset):
	def __init__(self, sequences, labels, max_len=100, pad_idx=0):
		self.sequences = sequences
		self.labels = labels
		self.max_len = max_len
		self.pad_idx = pad_idx

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, idx):
		seq = self.sequences[idx][: self.max_len]
		if not seq:
			seq = [self.pad_idx]

		length = len(seq)
		padded = seq + [self.pad_idx] * (self.max_len - length)

		return (
			torch.tensor(padded, dtype=torch.long),
			torch.tensor(length, dtype=torch.long),
			torch.tensor(self.labels[idx], dtype=torch.long),
		)


def build_torch_dataloader(samples, word_to_idx, batch_size=32, max_len=256, shuffle=False):
	sequences = [encode_text(sample["text"], word_to_idx) for sample in samples]
	labels = [sample["model"] for sample in samples]

	dataset = TextDataset(
		sequences=sequences,
		labels=labels,
		max_len=max_len,
		pad_idx=word_to_idx[PAD_TOKEN],
	)

	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class GRUClassifier(nn.Module):
	def __init__(
		self,
		embedding_matrix: np.ndarray,
		num_classes: int,
		hidden_size: int = 128,
		num_layers: int = 1,
		bidirectional: bool = True,
		dropout: float = 0.3,
		freeze_embeddings: bool = False,
		padding_idx: int = 0,
	):
		super().__init__()

		_, embedding_dim = embedding_matrix.shape

		self.embedding = nn.Embedding.from_pretrained(
			embeddings=torch.tensor(embedding_matrix, dtype=torch.float32),
			freeze=freeze_embeddings,
			padding_idx=padding_idx,
		)

		self.gru = nn.GRU(
			input_size=embedding_dim,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			bidirectional=bidirectional,
			dropout=dropout if num_layers > 1 else 0.0,
		)

		gru_output_dim = hidden_size * 2 if bidirectional else hidden_size

		self.dropout = nn.Dropout(dropout)
		self.classifier = nn.Sequential(
			nn.Linear(gru_output_dim, 64),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(64, num_classes),
		)

	def forward(self, x, lengths):
		embedded = self.embedding(x)

		packed = nn.utils.rnn.pack_padded_sequence(
			embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
		)

		_, hidden = self.gru(packed)

		if self.gru.bidirectional:
			hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
		else:
			hidden = hidden[-1]

		hidden = self.dropout(hidden)
		logits = self.classifier(hidden)
		return logits


def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
	model.train()
	total_loss = 0.0
	total_correct = 0
	total_examples = 0

	for inputs, lengths, labels in dataloader:
		inputs = inputs.to(device)
		lengths = lengths.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		logits = model(inputs, lengths)
		loss = criterion(logits, labels)
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
		optimizer.step()

		batch_size = labels.size(0)
		total_loss += loss.item() * batch_size
		total_correct += (logits.argmax(dim=1) == labels).sum().item()
		total_examples += batch_size

	return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_examples = 0

	for inputs, lengths, labels in dataloader:
		inputs = inputs.to(device)
		lengths = lengths.to(device)
		labels = labels.to(device)

		logits = model(inputs, lengths)
		loss = criterion(logits, labels)

		batch_size = labels.size(0)
		total_loss += loss.item() * batch_size
		total_correct += (logits.argmax(dim=1) == labels).sum().item()
		total_examples += batch_size

	return total_loss / total_examples, total_correct / total_examples


def run_quick_experiment(
	record_path="datasets/records.json",
	size=50000,
	split=(70, 20, 10),
	batch_size=32,
	max_len=256,
	embedding_dim=300,
	hidden_size=128,
	epochs=5,
	lr=1e-3,
	weight_decay=1e-4,
	min_freq=2,
	max_vocab_size=20000,
	early_stopping_patience=8,
	early_stopping_min_delta=1e-4,
	use_class_weights=True,
	seed=42,
):
	np.random.seed(seed)
	torch.manual_seed(seed)

	dm = SentenceDataModule(record_path=record_path, size=size, split=split, seed=seed)

	train_samples = dm.get_train_loader().samples
	val_samples = dm.get_val_loader().samples
	test_samples = dm.get_test_loader().samples

	vocabulary, word_to_idx = build_vocabulary(
		train_samples,
		min_freq=min_freq,
		max_vocab_size=max_vocab_size,
	)

	embedding_matrix = build_embedding_matrix(
		vocabulary=vocabulary,
		embedding_dim=embedding_dim,
		seed=seed,
	)

	train_loader = build_torch_dataloader(
		train_samples,
		word_to_idx,
		batch_size=batch_size,
		max_len=max_len,
		shuffle=True,
	)
	val_loader = build_torch_dataloader(
		val_samples,
		word_to_idx,
		batch_size=batch_size,
		max_len=max_len,
		shuffle=False,
	)
	test_loader = build_torch_dataloader(
		test_samples,
		word_to_idx,
		batch_size=batch_size,
		max_len=max_len,
		shuffle=False,
	)

	num_classes = len({sample["model"] for sample in train_samples + val_samples + test_samples})
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = GRUClassifier(
		embedding_matrix=embedding_matrix,
		num_classes=num_classes,
		hidden_size=hidden_size,
		padding_idx=word_to_idx[PAD_TOKEN],
	).to(device)

	if use_class_weights:
		class_counts = Counter(sample["model"] for sample in train_samples)
		weights = []
		for class_idx in range(num_classes):
			count = class_counts.get(class_idx, 0)
			weights.append(1.0 / max(count, 1))
		class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
		class_weights = class_weights / class_weights.sum() * num_classes
		criterion = nn.CrossEntropyLoss(weight=class_weights)
	else:
		criterion = nn.CrossEntropyLoss()

	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=0.5,
		patience=max(2, early_stopping_patience // 2),
	)

	print(f"Training GRU on {device} with vocab size {len(vocabulary)}")
	best_val_loss = float("inf")
	best_epoch = 0
	epochs_without_improvement = 0
	best_state_dict = {
		name: param.detach().cpu().clone()
		for name, param in model.state_dict().items()
	}

	for epoch in range(epochs):
		train_loss, train_acc = train_one_epoch(
			model, train_loader, optimizer, criterion, device
		)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		scheduler.step(val_loss)

		print(
			f"Epoch {epoch + 1:03d} | "
			f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
			f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
			f"lr={optimizer.param_groups[0]['lr']:.2e}"
		)

		if val_loss < (best_val_loss - early_stopping_min_delta):
			best_val_loss = val_loss
			best_epoch = epoch + 1
			epochs_without_improvement = 0
			best_state_dict = {
				name: param.detach().cpu().clone()
				for name, param in model.state_dict().items()
			}
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= early_stopping_patience:
				print(
					f"Early stopping at epoch {epoch + 1:03d} "
					f"(best epoch {best_epoch:03d}, best val_loss={best_val_loss:.4f})"
				)
				break

	model.load_state_dict(best_state_dict)
	print(
		f"Loaded best checkpoint from epoch {best_epoch:03d} "
		f"with val_loss={best_val_loss:.4f}"
	)

	test_loss, test_acc = evaluate(model, test_loader, criterion, device)
	print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")

	return {
		"model": model,
		"word_to_idx": word_to_idx,
		"vocabulary": vocabulary,
		"best_epoch": best_epoch,
		"best_val_loss": best_val_loss,
		"test_loss": test_loss,
		"test_acc": test_acc,
	}


if __name__ == "__main__":
	run_quick_experiment()
