import datasets
    

def get_dataset(size=None):
    """
    Load the dataset.
    """
    if size is None:
        train_dataset = datasets.load_dataset(
            'wikipedia',
            '20220301.en',
            split='train'
        )
    else:
        train_dataset = datasets.load_dataset(
            'wikipedia',
            split='train',
            streaming=True
        )
        train_dataset = train_dataset.take(size)
    return train_dataset


if __name__ == "__main__":
    get_dataset()