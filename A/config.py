from argparse import Namespace
from pathlib import Path

PARENT_FOLDER = Path().absolute()

# Args:the following code selects a matscibert model and corresponding training parameters
args = dict(
    model_name="matscibert",  # or 'bert'
    model_save_dir=str(PARENT_FOLDER),
    cache_dir=str(PARENT_FOLDER),
    num_epochs=3,
    batch_size=16,
    seeds=11,
    lm_lrs=4e-4,
    lr_scheduler_type="cosine_with_restarts",
    weight_decay=0.01,
    dataset_name="fatigue",
    model_revision="main",
    max_seq_length=512
)

args = Namespace(**args)
