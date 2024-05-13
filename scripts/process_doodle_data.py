from datasets import load_dataset
import random
# https://huggingface.co/datasets/clint-greene/doodles-captions

def get_doodle_data():
    dataset = load_dataset("julianmoraes/doodles-captions-BLIP", revision='main', split='train')
    dataset = dataset.shuffle(seed=42)

    train_size = int(len(dataset) * 0.8)
    test_size = int(len(dataset) * 0.1)

    train = dataset.select(range(train_size))
    test = dataset.select(range(train_size, train_size + test_size))
    val = dataset.select(range(train_size + test_size, len(dataset)))
    print(len(train))




if __name__ == "__main__":
    get_doodle_data()