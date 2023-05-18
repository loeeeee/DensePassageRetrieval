import json

def read_source(size: str = "1000"):
    with open(f"./../.data/example_embeddings_{size}.json", "r") as f:
        source = json.load(f)
    train_split = int(len(source) * 0.9)
    train = source[:train_split]
    test = source[train_split:]
    return train, test
