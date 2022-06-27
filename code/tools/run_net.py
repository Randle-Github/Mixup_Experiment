"""Wrapper to train and test a video classification model."""
import sys
sys.path.insert(0, '.') # assume we run from parent dir
from compaction.utils.misc import launch_job
from compaction.utils.parser import load_config, parse_args

from test_net import test
from train_net import train


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
