from A.get_dataset import get_data


def load_dataset(dataset_name: str, subset: str = "CGNER"):
    """
    Load train, validation, and test data from the dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., "fatigue").
        subset (str): Either "CGNER" or "FGNER".

    Returns:
        tuple: (train, dev, test) where each is a Namespace with .tokens and .tags
    """
    train, dev, test = get_data(dataset_name, subset=subset)
    print(f"Loaded dataset '{dataset_name}' with subset '{subset}'")
    print(f"Train samples: {len(train.tokens)}, Validation samples: {len(dev.tokens)}, Test samples: {len(test.tokens)}")
    return train, dev, test


def get_label_list_from_tags(tag_sequences):
    """
    Get sorted list of unique labels from a list of tag sequences.

    Args:
        tag_sequences (List[List[str]]): Nested list of label tags for each sentence.

    Returns:
        List[str]: Sorted list of unique labels.
    """
    unique_labels = set(label for sentence in tag_sequences for label in sentence)
    return sorted(list(unique_labels))


def build_tag_maps(tag_sequences):
    """
    Build tag2id and id2tag mappings from label sequences.

    Args:
        tag_sequences (List[List[str]]): Nested list of label tags.

    Returns:
        Tuple: (label_list, tag2id, id2tag)
    """
    label_list = get_label_list_from_tags(tag_sequences)
    tag2id = {tag: idx for idx, tag in enumerate(label_list)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    print(f"Number of unique labels: {len(label_list)}")
    return label_list, tag2id, id2tag
