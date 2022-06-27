from os import get_blocking
import pickle
import torch
import numpy as np
import pprint
import os

from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from compaction.model import build_model
from compaction.model.mixup_helper import mixup_process
import compaction.utils.logging as logging
import compaction.model.optimizer as optim
import compaction.utils.checkpoint as cu
import compaction.utils.distributed as du
import compaction.utils.metrics as metrics
import compaction.model.losses as losses
import compaction.visualization.tensorboard_vis as tb
from compaction.datasets import loader
from compaction.utils.meters import MultiLossTrainMeter, MultiLossValMeter, TrainMeter, ValMeter, EPICTrainMeter, EPICValMeter
from compaction.utils.wechat_bot import WeChat
import compaction.utils.misc as misc
from compaction.model.mixup_helper import MixUp
# from timm.utils import NativeScaler  # 这是干嘛的？
from compaction.utils.timm_scaler import NativeScaler

logger = logging.get_logger(__name__)
wx = WeChat()

def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg,
                writer=None, mixup_fn=None, loss_fun=None, loss_scaler=None):

    # Enable train mode.
    model.train()

    train_meter.iter_tic()
    data_size = len(train_loader)
    for cur_iter, (inputs, labels, index, meta) in enumerate(train_loader):
        # meta 用于存放杂项的标注内容
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if not isinstance(val[i], (str,)):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)

        # 第一个 optimizer 是 box, 第二个是 vis
        if isinstance(optimizer, list):
            for i, opt in enumerate(optimizer):
                if i == 0:
                    optim.set_lr(opt, cfg.SOLVER.BOX_LR) # box 的 lr 一直固定不变
                else:
                    optim.set_lr(opt, lr)
        else:
            optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.SOLVER.USE_MIXED_PRECISION):
            if 'region' in cfg.MODEL.HEAD:
                preds = model(inputs, meta, labels)
            else:
                preds = model(inputs, labels)

            global_step = data_size * cur_epoch + cur_iter
            if cfg.TRAIN.MIXUP and cfg.TRAIN.MULTI_LOSS:
                loss_fuse = loss_fun(preds[0], labels)
                loss_vis = loss_fun(preds[1], labels)
                loss_box = loss_fun(preds[2], labels)
                comp_cls, mix_cls, mix_label = preds[-3], preds[-2], preds[-1]
                loss_comp = loss_fun(comp_cls, labels)
                soft_loss = losses.get_loss_func("soft_target_cross_entropy")()
                loss_mix = soft_loss(mix_cls, mix_label)
                loss = loss_fuse + loss_vis + loss_box + 0.5* (loss_mix + loss_comp)
                # loss = loss_fuse + loss_vis + loss_box + loss_mix
            elif cfg.TRAIN.MIXUP:
                comp_cls, mix_cls, mix_label = preds[-3], preds[-2], preds[-1]
                # mix_cls, mix_label = preds[-2], preds[-1]
                loss = loss_fun(preds[0], labels)
                # loss_comp = loss_fun(comp_cls, labels)
                soft_loss = losses.get_loss_func("soft_target_cross_entropy")()
                loss_mix = soft_loss(mix_cls, mix_label)
                loss_comp = loss_fun(comp_cls, labels)
                loss = loss + 0.5 * (loss_mix + loss_comp)
                # loss = loss + loss_mix
            elif isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "Epickitchens":
                # Compute the loss.
                loss_verb = loss_fun(preds[0], labels['verb'])
                loss_noun = loss_fun(preds[1], labels['noun'])
                loss = 0.5 * (loss_verb + loss_noun)
            elif cfg.TRAIN.MULTI_LOSS:
                loss_fuse = loss_fun(preds[0], labels)
                loss_vis = loss_fun(preds[1], labels)
                loss_box = loss_fun(preds[2], labels)
                # loss = [loss_fuse + loss_vis, loss_box]
                loss = loss_fuse + loss_vis + loss_box
            else:
                if cfg.MODEL.HEAD in ['region_composer']:
                    cls, comp_cls, comp_label = preds
                    preds = cls
                    soft_loss = losses.get_loss_func("soft_target_cross_entropy")()
                    loss = loss_fun(cls, labels)
                    loss_comp = soft_loss(comp_cls, comp_label)
                    loss = loss + loss_comp
                elif cfg.MODEL.HEAD == 'region_multitask' or 'multitask' in cfg.OUTPUT_DIR:
                    act_cls, obj_cls, obj_labels = preds
                    act_loss = loss_fun(act_cls, labels)
                    obj_loss = loss_fun(obj_cls, obj_labels)
                    loss = act_loss + obj_loss
                else:
                    loss = loss_fun(preds, labels)

            # check Nan Loss.
            if isinstance(loss, list):
                for l in loss:
                    misc.check_nan_losses(l)
            else:
                misc.check_nan_losses(loss)

            # Perform the backward pass.
            if isinstance(optimizer, list):
                for opt in optimizer:
                    opt.zero_grad()
            else:
                optimizer.zero_grad()

            if cfg.SOLVER.USE_MIXED_PRECISION: # Mixed Precision Training
                if isinstance(optimizer, list):
                    is_second_order = hasattr(optimizer[0], 'is_second_order') and optimizer.is_second_order
                else:
                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=cfg.SOLVER.CLIP_GRAD,
                            parameters=model.parameters(), create_graph=is_second_order) #NOTE: 关注这个东西
            else:
                loss.backward()
                # Update the parameters.
                if isinstance(optimizer, list):
                    for opt in optimizer:
                        opt.step()
                    else:
                        optimizer.step()

            # log acc
            top1_err, top5_err = None, None
            if cfg.TRAIN.MULTI_LOSS and not cfg.DATA.MULTI_LABEL:
                # Compute the verb accuracies.
                fuse_top1_acc, fuse_top5_acc = metrics.topk_accuracies(
                    preds[0], labels, (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_fuse, fuse_top1_acc, fuse_top5_acc = du.all_reduce(
                        [loss_fuse, fuse_top1_acc, fuse_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_fuse, fuse_top1_acc, fuse_top5_acc = (
                    loss_fuse.item(),
                    fuse_top1_acc.item(),
                    fuse_top5_acc.item(),
                )

                # Compute the noun accuracies.
                vis_top1_acc, vis_top5_acc = metrics.topk_accuracies(
                    preds[1], labels, (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_vis, vis_top1_acc, vis_top5_acc = du.all_reduce(
                        [loss_vis, vis_top1_acc, vis_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_vis, vis_top1_acc, vis_top5_acc = (
                    loss_vis.item(),
                    vis_top1_acc.item(),
                    vis_top5_acc.item(),
                )

                box_top1_acc, box_top5_acc = metrics.topk_accuracies(
                    preds[2], labels, (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_box, box_top1_acc, box_top5_acc = du.all_reduce(
                        [loss_box, box_top1_acc, box_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_box, box_top1_acc, box_top5_acc = (
                    loss_box.item(),
                    box_top1_acc.item(),
                    box_top5_acc.item(),
                )

                # Update and log stats.
                train_meter.update_stats(
                    (fuse_top1_acc, vis_top1_acc, box_top1_acc),
                    (fuse_top5_acc, vis_top5_acc, box_top5_acc),
                    (loss_fuse, loss_vis, loss_box),
                    lr, inputs[0].size(0) * cfg.NUM_GPUS
                )
            elif cfg.DATA.MULTI_LABEL:  # charades
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
                # Update and log stats.
                train_meter.update_stats(
                    top1_err,
                    top5_err,
                    loss,
                    lr,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),
                )
            else:
                if cfg.TRAIN.MIXUP or cfg.MODEL.HEAD == 'region_multitask' or isinstance(preds, tuple): # hack
                    preds = preds[0]
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )
                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )
                # Update and log stats.
                train_meter.update_stats(
                    top1_err,
                    top5_err,
                    loss,
                    lr,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),
                )

                # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )
                if cfg.TRAIN.MULTI_LOSS:
                    writer.add_scalars(
                        {
                            "Train/fuse_top1_acc": fuse_top1_acc,
                            "Train/fuse_top5_acc": fuse_top5_acc,
                            "Train/vis_top1_acc": vis_top1_acc,
                            "Train/vis_top5_acc": vis_top5_acc,
                            "Train/box_top1_acc": box_top1_acc,
                            "Train/box_top5_acc": box_top5_acc,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                        )
                else:
                    writer.add_scalars(
                        {
                            "Train/Top1_err": top1_err if top1_err is not None else 0.0,
                            "Train/Top5_err": top5_err if top5_err is not None else 0.0,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if not isinstance(val[i], (str,)):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.SOLVER.USE_MIXED_PRECISION):
            if 'region' in cfg.MODEL.HEAD:
                preds = model(inputs, meta, labels)
            else:
                preds = model(inputs, labels)

            if cfg.TRAIN.MULTI_LOSS and not cfg.DATA.MULTI_LABEL:
                # Compute the verb accuracies.
                fuse_top1_acc, fuse_top5_acc = metrics.topk_accuracies(
                    preds[0], labels, (1, 5))

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    fuse_top1_acc, fuse_top5_acc = du.all_reduce(
                        [fuse_top1_acc, fuse_top5_acc]
                    )

                # Copy the errors from GPU to CPU (sync point).
                fuse_top1_acc, fuse_top5_acc = (
                    fuse_top1_acc.item(),
                    fuse_top5_acc.item(),
                )

                # Copy the errors from GPU to CPU (sync point).
                vis_top1_acc, vis_top5_acc = metrics.topk_accuracies(
                    preds[1], labels, (1, 5))

                if cfg.NUM_GPUS > 1:
                    vis_top1_acc, vis_top5_acc = du.all_reduce(
                        [vis_top1_acc, vis_top5_acc]
                    )

                # Copy the errors from GPU to CPU (sync point).
                vis_top1_acc, vis_top5_acc = (
                    vis_top1_acc.item(),
                    vis_top5_acc.item(),
                )

                # Compute the action accuracies.
                box_top1_acc, box_top5_acc = metrics.topk_accuracies(
                    preds[2], labels, (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    box_top1_acc, box_top5_acc = du.all_reduce(
                        [box_top1_acc, box_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                box_top1_acc, box_top5_acc = (
                    box_top1_acc.item(),
                    box_top5_acc.item(),
                )

                val_meter.iter_toc()

                # Update and log stats.
                val_meter.update_stats(
                    (fuse_top1_acc, vis_top1_acc, box_top1_acc),
                    (fuse_top5_acc, vis_top5_acc, box_top5_acc),
                    inputs[0].size(0) * cfg.NUM_GPUS
                )

                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {
                            "Val/fuse_top1_acc": fuse_top1_acc,
                            "Val/fuse_top5_acc": fuse_top5_acc,
                            "Val/vis_top1_acc": vis_top1_acc,
                            "Val/vis_top5_acc": vis_top5_acc,
                            "Val/box_top1_acc": box_top1_acc,
                            "Val/box_top5_acc": box_top5_acc,
                        },
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )
            elif cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else: # 不是 epic-kitchen 这个数据集
                # Compute the errors.
                if isinstance(preds, tuple):
                    preds = preds[0]
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            if cfg.DATA.MULTI_LABEL and cfg.TRAIN.MULTI_LOSS:
                preds = preds[0]
            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [
            label.clone().detach() for label in val_meter.all_labels
        ]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(
            preds=all_preds, labels=all_labels, global_step=cur_epoch
        )
    # send to wechat
    if du.is_master_proc(du.get_world_size()):
        if cfg.TRAIN.MULTI_LOSS and not cfg.DATA.MULTI_LABEL:
            fuse_top1_acc = val_meter.num_fuse_top1_cor / val_meter.num_samples
            fuse_top5_acc = val_meter.num_fuse_top5_cor / val_meter.num_samples
            vis_top1_acc = val_meter.num_vis_top1_cor / val_meter.num_samples
            vis_top5_acc = val_meter.num_vis_top5_cor / val_meter.num_samples
            box_top1_acc = val_meter.num_box_top1_cor / val_meter.num_samples
            box_top5_acc = val_meter.num_box_top5_cor / val_meter.num_samples
            exp_name = cfg.OUTPUT_DIR.split("/")[-1]
            message = (f'exp: {exp_name}\nepoch: {cur_epoch}\n'
                       f'fuse acc1: {fuse_top1_acc:.2f}\n'
                       f'fuse acc5: {fuse_top5_acc:.2f}\n'
                       f'max fuse 1: {val_meter.max_fuse_top1_acc:.2f}\n'
                       f'max fuse 5: {val_meter.max_fuse_top5_acc:.2f}\n'
                       f'vis acc1: {vis_top1_acc:.2f}\n'
                       f'box acc1: {box_top1_acc:.2f}\n')
            wx.send_message(message)
        elif cfg.TRAIN.DATASET == 'Actiongenome':
            pass
        else:
            top1_err = val_meter.num_top1_mis / val_meter.num_samples
            top5_err = val_meter.num_top5_mis / val_meter.num_samples
            min_top1_err = val_meter.min_top1_err
            min_top5_err = val_meter.min_top5_err
            exp_name = cfg.OUTPUT_DIR.split("/")[-1]
            message = f'exp: {exp_name}\nepoch: {cur_epoch}\nAcc1: {100-top1_err:.2f}\nAcc5: {100-top5_err:.2f}\nmaxAcc1: {100-min_top1_err:.2f}\nmaxAcc5: {100-min_top5_err:.2f}'
            wx.send_message(message)

    val_meter.reset()

def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    #NOTE: 具体有什么改进？
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)

def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO and cfg.DATA.INPUT_TYPE == 'rgb':
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    # optimizer = optim.construct_optimizer(model, cfg)
    optimizer = optim.custom_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    去掉了 multigrid 的部分
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    if not cfg.SOLVER.CUSTOM_OPTIM:
        optimizer = optim.construct_optimizer(model, cfg)
    else:
        optimizer = optim.custom_optimizer(model, cfg)

    # Mixed Precision Training Scaler
    if cfg.SOLVER.USE_MIXED_PRECISION:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, loss_scaler=loss_scaler)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.TRAIN.MULTI_LOSS and not cfg.DATA.MULTI_LABEL:
        train_meter = MultiLossTrainMeter(len(train_loader), cfg)
        val_meter = MultiLossValMeter(len(val_loader), cfg)
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    mixup_fn = None
    mixup_active = cfg.MIXUP.MIXUP_ALPHA > 0 or cfg.MIXUP.CUTMIX_ALPHA > 0 or cfg.MIXUP.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.MIXUP_ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            switch_prob=cfg.MIXUP.MIXUP_SWITCH_PROB,
            label_smoothing=cfg.SOLVER.SMOOTHING,
            num_classes=cfg.MODEL.NUM_CLASSES
        )

    # Explicitly declare reduction to mean.
    if cfg.MIXUP.MIXUP_ALPHA > 0.:
        # smoothing is handled with mixup label transform
        loss_fun = losses.get_loss_func("soft_target_cross_entropy")()
    elif cfg.SOLVER.SMOOTHING > 0.0:
        loss_fun = losses.get_loss_func("label_smoothing_cross_entropy")(
            smoothing=cfg.SOLVER.SMOOTHING)
    else:
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    if cfg.TRAIN.CUSTOM_SAMPLER:        
        with open(os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'object_relevant', 'samplesA.pkl'), 'rb') as f:
            seqA = pickle.load(f)
        with open(os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'object_relevant', 'samplesB.pkl'), 'rb') as f:
            seqB = pickle.load(f)

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if not cfg.DEBUG:
            if cfg.TRAIN.CUSTOM_SAMPLER:
                seq = (seqA[cur_epoch], seqB[cur_epoch])
                train_loader = loader.my_construct_train_loader(cfg, seq)
            else:
                # Shuffle the dataset.
                loader.shuffle_dataset(train_loader, cur_epoch)

            # Train for one epoch.
            train_epoch(
                train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer,
                loss_scaler=loss_scaler, loss_fun=loss_fun, mixup_fn=mixup_fn)

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None
        )
        if cfg.DEBUG:
            is_checkp_epoch = True
            is_eval_epoch = True

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg,
                loss_scaler=loss_scaler)
        # TODO: save lateset epoch model
        cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, loss_scaler=loss_scaler, latest=True)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
