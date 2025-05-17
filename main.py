import argparse
import torch
import os
from functools import partial

from A import dataloader, model, trainer, utils, get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train NER model with Transformers")
    parser.add_argument("--model_name", type=str, default="matscibert",
                        choices=["matscibert", "bert"], help="Model alias to use")
    parser.add_argument("--dataset_name", type=str, default="fatigue",
                        help="Name of dataset in get_dataset")
    parser.add_argument("--subset", type=str, default="CGNER",
                        choices=["CGNER", "FGNER"], help="Which subset of data to use")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lm_lrs", type=float, default=4e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_restarts",
                        help="Learning rate scheduler type")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum input sequence length")
    parser.add_argument("--seeds", type=int, default=11,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory to save models")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory to cache model/tokenizer")
    parser.add_argument("--revision", type=str, default="main",
                        help="Model branch or revision")
    


    return parser.parse_args()

def main():
    args = parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Make sure output and cache dirs exist
    output_dir = utils.ensure_dir(os.path.join(args.output_dir, args.model_name))
    cache_dir = utils.ensure_dir(args.cache_dir)

    # Load dataset
    train, dev, test = dataloader.load_dataset(args.dataset_name, args.subset)
    label_list, tag2id, id2tag = dataloader.build_tag_maps(train.tags)

    # Load model/tokenizer/config
    model_instance, tokenizer, _ = model.load_model_and_tokenizer(
        model_alias=args.model_name,
        num_labels=len(label_list),
        revision=args.revision,
        cache_dir=cache_dir
    )
    model_instance = model_instance.to(device)

    # Create tokenized datasets
    train_dataset, val_dataset, test_dataset = utils.create_dataset(
        tokenizer, train, dev, test,
        max_seq_length=args.max_seq_length,
        tag2id=tag2id,
        dataset_name=args.dataset_name
    )

    # Build trainer
    trainer_instance = trainer.build_trainer(
        model=model_instance,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args,
        output_dir=output_dir,
        compute_metrics_fn=partial(utils.compute_metrics, id2tag=id2tag)
    )

    # Train
    trainer_instance.train()

    # Evaluate on validation set
    val_result = trainer_instance.evaluate()
    print("Validation Result:", val_result)

    # Evaluate on test set
    test_result = trainer_instance.evaluate(test_dataset, metric_key_prefix="test")
    print("Test Result:", test_result)

    # Output predictions
    test_predictions, test_labels, _ = trainer_instance.predict(test_dataset)
    utils.generate_and_save_report(
    predictions=test_predictions,
    labels=test_labels,
    label_list=label_list,
    output_path="results/test_classification_report.csv"
    )   


if __name__ == "__main__":
    main()