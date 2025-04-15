from datasets import load_dataset, train_test_split

def sample(n=100):
    ds = load_dataset("FiscalNote/billsum")
    sample = ds.train_test_split(test_size=n, seed=42)["test"]
    return sample.to_pandas()
