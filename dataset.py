import datasets
    

def get_dataset(dataset, subset, size=None):
    """
    Load the dataset.
    """
    if size is None:
        train_dataset = datasets.load_dataset(
            dataset,
            subset
        )
    else:
        train_dataset = datasets.load_dataset(
            dataset,
            subset,
            streaming=True
        )
        train_dataset = train_dataset.take(size)
    return train_dataset


if __name__ == "__main__":
    train_dataset = get_dataset('glue', 'mrpc')
    print(train_dataset)
    for i in train_dataset:
        print(i)
        break