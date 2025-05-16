from transformers import Trainer, TrainingArguments, AdamW


def build_optimizer(model, learning_rate: float, weight_decay: float):
    """
    Create AdamW optimizer for the model.

    Args:
        model: HuggingFace model
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay

    Returns:
        torch.optim.Optimizer: AdamW optimizer
    """
    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in model.named_parameters()],
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer


def build_training_arguments(args, output_dir: str):
    """
    Build HuggingFace TrainingArguments from the config.

    Args:
        args (Namespace): Configuration parameters
        output_dir (str): Output directory for saving checkpoints

    Returns:
        TrainingArguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        warmup_ratio=0.1,
        weight_decay=args.weight_decay,
        learning_rate=args.lm_lrs,
        seed=args.seeds,
        lr_scheduler_type=args.lr_scheduler_type,
    )


def build_trainer(model, tokenizer, train_dataset, val_dataset,
                  args, output_dir, compute_metrics_fn):
    """
    Build HuggingFace Trainer instance.

    Args:
        model: The model to train
        tokenizer: Tokenizer for padding/collation
        train_dataset: Tokenized train dataset
        val_dataset: Tokenized validation dataset
        args (Namespace): Config
        output_dir (str): Where to save checkpoints
        compute_metrics_fn (Callable): Metrics function using id2tag

    Returns:
        Trainer
    """
    training_args = build_training_arguments(args, output_dir)
    optimizer = build_optimizer(model, args.lm_lrs, args.weight_decay)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        optimizers=(optimizer, None),  # use Trainer scheduler
    )

    return trainer
