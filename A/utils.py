from pathlib import Path
import numpy as np
import A.get_dataset
from A.get_dataset import MyDataset
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from seqeval.metrics import classification_report
import numpy as np
import os
import pandas as pd



def ensure_dir(dir_path):
    """
    Ensure the directory given in the command line arguments exists
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


def remove_zero_len_tokens(X, y, tokenizer, dataset_name):
    """
    Remove zero length tokens from the dataset.
    @param X - the input sentences
    @param y - the corresponding labels
    @param tokenizer - the tokenizer used to tokenize the sentences
    @param dataset_name - the name of the dataset
    @return new_X - the input sentences with zero length tokens removed
    @return new_y - the corresponding labels with zero length tokens removed

    """

    new_X, new_y = [], []
    for sent, labels in zip(X, y):
        new_sent, new_labels = [], []
        for token, label in zip(sent, labels):
            if len(tokenizer.tokenize(token)) == 0:
                assert dataset_name == "fatigue"
                continue
            new_sent.append(token)
            new_labels.append(label)
        new_X.append(new_sent)
        new_y.append(new_labels)
    return new_X, new_y


def encode_tags(tags, encodings, tag2id):
    """
    Encode the tags for the dataset using a tag-to-id mapping and the token
    encodings.
    @param tags - the tags to be encoded
    @param encodings - the token encodings for the dataset
    @param tag2id - a dictionary mapping tags to their corresponding ids
    @return encoded_labels - a list of encoded labels
    """

    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        non_trun_ele = ((arr_offset[:, 0] == 0) &
                        (arr_offset[:, 1] != 0)).sum()
        doc_enc_labels[(arr_offset[:, 0] == 0) &
                       (arr_offset[:, 1] != 0)] = doc_labels[
            :non_trun_ele
        ]
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


def create_dataset(tokenizer, train, dev, test, max_seq_length,
                   tag2id, dataset_name):
    """
    Create a dataset for training, validation, and testing.
    @param tokenizer - the tokenizer used to tokenize the input data
    @param train - the training dataframe
    @param dev - the validation dataframe
    @param test - the testing dataframe
    @param max_seq_length - the maximum sequence length for the input data
    @param tag2id - a dictionary mapping tags to their corresponding IDs
    @param dataset_name - the name of the dataset

    """

    train_X, train_y = train.tokens, train.tags
    val_X, val_y = dev.tokens, dev.tags
    test_X, test_y = test.tokens, test.tags

    train_X, train_y = remove_zero_len_tokens(train_X, train_y,
                                              tokenizer, dataset_name)
    val_X, val_y = remove_zero_len_tokens(val_X, val_y, tokenizer,
                                          dataset_name)
    test_X, test_y = remove_zero_len_tokens(test_X, test_y, tokenizer,
                                            dataset_name)

    train_encodings = tokenizer(
        train_X,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    )
    val_encodings = tokenizer(
        val_X,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    )
    test_encodings = tokenizer(
        test_X,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    )

    train_labels = encode_tags(train_y, train_encodings, tag2id)
    val_labels = encode_tags(val_y, val_encodings, tag2id)
    test_labels = encode_tags(test_y, test_encodings, tag2id)

    train_encodings.pop("offset_mapping")
    val_encodings.pop("offset_mapping")
    test_encodings.pop("offset_mapping")

    train_dataset = MyDataset(train_encodings, train_labels)
    val_dataset = MyDataset(val_encodings, val_labels)
    test_dataset = MyDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset


def compute_metrics(p, id2tag):
    """
    Compute various metrics for a token classification task.
    @param p - a tuple containing the predictions and labels
    @param id2tag - a dictionary mapping tag IDs to tag names
    @return a dictionary containing the computed metrics: f1,
    recall, precision, and accuracy
    """

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (_, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    preds, labs = [], []
    for pred, lab in zip(true_predictions, true_labels):
        preds.extend(pred)
        labs.extend(lab)
    assert len(preds) == len(labs)

    results = {}

    results["f1"] = f1_score(true_labels, true_predictions)
    results["recall"] = recall_score(true_labels, true_predictions)
    results["precision"] = precision_score(true_labels, true_predictions)
    results["accuracy"] = accuracy_score(true_labels, true_predictions)

    return results

def generate_and_save_report(predictions, labels, label_list, output_path=None):
    # Convert predicted label indices to tag strings, ignoring padding (-100)
    pred_tags = [
        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(np.argmax(predictions, axis=2), labels)
    ]
    true_tags = [
        [label_list[l] for (_, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(np.argmax(predictions, axis=2), labels)
    ]

    # Sanity check: ensure lengths of predictions and labels match
    assert all(len(p) == len(t) for p, t in zip(pred_tags, true_tags))

    print("Test classification report")
    report_str = classification_report(true_tags, pred_tags)
    print(report_str)

    # If output_path is specified, save the detailed report as a CSV file
    if output_path:
        report_dict = classification_report(true_tags, pred_tags, output_dict=True)
        df = pd.DataFrame(report_dict).transpose()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=True)
        print(f"Detailed classification report saved to {output_path}")