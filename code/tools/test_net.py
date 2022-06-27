import numpy as np
import os
import pickle
import torch
from torch.overrides import wrap_torch_function

import compaction.utils.checkpoint as cu
import compaction.utils.distributed as du
import compaction.utils.logging as logging
import compaction.utils.misc as misc
import compaction.model.optimizer as optim

from compaction.datasets import loader
from compaction.model import build_model
from compaction.utils.meters import TestMeter

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if 'region' in cfg.MODEL.HEAD:
            preds = model(inputs, meta, labels)
        else:
            preds = model(inputs, labels)

        if isinstance(preds, tuple):
            preds = preds[0]

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx]
            )
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    all_preds = test_meter.video_preds.clone().detach()
    all_labels = test_meter.video_labels

    if cfg.NUM_GPUS:
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu()

    if cfg.TEST.SAVE_RESULTS_PATH != "":
        save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

        if du.is_root_proc():
            with open(save_path, "wb") as f:
                pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics()

    return test_meter
            
def test(cfg):
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)

    logger.info("Test with config:")
    logger.info(cfg)

    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)
    # optimizer = optim.custom_optimizer(model, cfg)
    # cu.load_train_checkpoint(cfg, model, optimizer)

    test_loader = loader.construct_loader(cfg, "val")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    
    test_meter = TestMeter(
        len(test_loader.dataset),
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg
        )

    writer = None
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    

    
