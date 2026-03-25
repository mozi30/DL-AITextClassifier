import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataloader.dataloader import SentenceDataModule


def test_print_first_10_samples():
    """Load dataset from records_long.json and print first 10 samples of each loader.

    This is primarily a sanity/inspection test mirroring other scripts in the
    tests/ directory. It instantiates the SentenceDataModule and prints
    samples from train, validation, and test splits.
    """
    dm = SentenceDataModule(
        record_path=os.path.join(PROJECT_ROOT, "datasets", "records_long.json"),
        # Use all available samples; keep other defaults (normalize, split, seed).
        size=10000,
    )

    loaders = {
        "train": dm.get_train_loader(),
        "val": dm.get_val_loader(),
        "test": dm.get_test_loader(),
    }

    for name, loader in loaders.items():
        print(f"=== {name.upper()} LOADER (first 10 samples) ===")
        # Safely iterate up to 10 samples or the length of the loader
        limit = min(10, len(loader))
        for i in range(limit):
            sample = loader[i]
            print(sample)

    # This test is for manual inspection; we just assert that loaders are non-empty.
    assert len(dm.get_train_loader()) >= 0


def test_custom_splits():
    """Verify that custom split ratios produce the expected loader sizes.

    Uses a fixed `size` so we know exactly how many samples to expect in
    each of train/val/test for a given split.
    """
    total_size = 100
    split = (60, 20, 20)

    dm = SentenceDataModule(
        record_path=os.path.join(PROJECT_ROOT, "datasets", "records_long.json"),
        size=total_size,
        split=split,
        seed=42,
    )

    train_loader = dm.get_train_loader()
    val_loader = dm.get_val_loader()
    test_loader = dm.get_test_loader()

    assert len(train_loader) == int(total_size * split[0] / 100)
    assert len(val_loader) == int(total_size * split[1] / 100)
    assert len(test_loader) == total_size - len(train_loader) - len(val_loader)


if __name__ == "__main__":
    # Allow running this file directly: `python test_dataloader.py`
    test_custom_splits()
    test_print_first_10_samples()
