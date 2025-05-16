import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append("..")

PARENT_DIR = Path(__file__).parent
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Datasets" / "data"


def flatten(line):
    """
    Flatten a nested list into a single list.
    @param line - the nested list to be flattened
    @return A flattened list
    """

    return [item for sublist in line for item in sublist]


def split_txt(line):
    """
    Split a line of text into a token and its associated tags.
    @param line - the line of text to be split
    @return A tuple containing the token and its tags
    """

    if line == "":
        return "", ""
    else:
        token, tags = line.split()
        return token, tags


def read_CoNLL2002_format(filename):
    """
    Read a file in the CoNLL-2002 format and return a pandas DataFrame.
    @param filename - the name of the file to read
    @return a pandas DataFrame containing the tokens and tags from the file
    """

    # read file
    with open(filename, encoding='utf-8') as f:
        lines = f.read().strip()

    # find sentence-like boundaries
    lines = lines.split("\n\n")

    # split on newlines
    lines = [line.split("\n") for line in lines]

    # get tokens
    tokens = [[split_txt(words)[0] for words in line] for line in lines]

    # get labels/tags
    tags = [[split_txt(words)[1] for words in line] for line in lines]

    # convert to df
    data = {"tokens": tokens, "tags": tags}
    df = pd.DataFrame(data=data)
    return df


def get_data(data_name, subset="CGNER"):
    """
    Retrieve the data based on the given dataset name.
    @param data_name - the name of the dataset to retrieve
    @param subset - CGNER or FGNER dataset
    @return The train, dev, and test data
    """

    data_name = data_name.lower()
    if subset == "CGNER":
        data_dir = DATA_DIR/"coarse_grained_training"
    elif subset == "FGNER":
        data_dir = DATA_DIR/"fine_grained_training"
    if data_name == "fatigue":
        train = read_CoNLL2002_format(
            os.path.join(data_dir, "train.txt"))
        print(
            "Train data: %d sentences, %d tokens"
            % (len(train), len(flatten(train.tokens)))
        )

        dev = read_CoNLL2002_format(os.path.join(data_dir, "val.txt"))
        print(
            "Dev data: %d sentences, %d tokens" % (
                len(dev), len(flatten(dev.tokens)))
        )

        test = read_CoNLL2002_format(os.path.join(data_dir, "test.txt"))
        print(
            "Test data: %d sentences, %d tokens"
            % (len(test), len(flatten(test.tokens)))
        )

    return train, dev, test


class MyDataset(torch.utils.data.Dataset):

    """
    A custom dataset class that inherits from `torch.utils.data.Dataset`.
    @param inp - The input data dictionary.
    @param tags - The target labels.
    @returns An instance of the dataset class.
    """

    def __init__(self, inp, tags):
        self.inp = inp
        self.tags = tags

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
        item["labels"] = torch.tensor(self.tags[idx])
        return item

    def __len__(self):
        return len(self.tags)
