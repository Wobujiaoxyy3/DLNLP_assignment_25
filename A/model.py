from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification


def get_model_name(model_alias: str) -> str:
    """
    Map a model alias (e.g. 'bert', 'matscibert') to its HuggingFace model name.

    Args:
        model_alias (str): Short name like 'bert' or 'matscibert'

    Returns:
        str: HuggingFace model ID
    """
    if model_alias == "matscibert":
        return "m3rg-iitd/matscibert"
    elif model_alias == "bert":
        return "google-bert/bert-base-uncased"
    else:
        raise ValueError(f"Unsupported model name: {model_alias}")


def load_model_and_tokenizer(model_alias: str,
                             num_labels: int,
                             revision: str,
                             cache_dir: str = None):
    """
    Load tokenizer, config, and token classification model.

    Args:
        model_alias (str): Alias like 'bert' or 'matscibert'
        num_labels (int): Number of classification labels
        revision (str): Model branch or revision
        cache_dir (str, optional): Directory for HuggingFace cache

    Returns:
        tuple: (model, tokenizer, config)
    """
    model_name = get_model_name(model_alias)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        revision=revision,
        cache_dir=cache_dir,
    )

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        revision=revision,
        cache_dir=cache_dir,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        revision=revision,
        cache_dir=cache_dir,
    )

    return model, tokenizer, config
