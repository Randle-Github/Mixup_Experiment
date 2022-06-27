"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import compaction.utils.logging as logging
import compaction.utils.misc as misc
import compaction.utils.metrics as metrics
logger = logging.get_logger(__name__)

class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count

class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_avg(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_avg()
            stats["top5_err"] = self.mb_top5_err.get_win_avg()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        logging.log_json_stats(stats)

class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = charades_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err

        logging.log_json_stats(stats)

# def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    # logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    # preds = preds[:, ~(np.all(labels == 0, axis=0))]
    # labels = labels[:, ~(np.all(labels == 0, axis=0))]
    # aps = [0]
    # try:
        # aps = average_precision_score(labels, preds, average=None)
    # except ValueError:
        # print(
            # "Average precision requires a sufficient number of samples \
            # in a batch which are missing in this sample."
        # )

    # mean_ap = np.mean(aps)
    # return mean_ap

def mAP(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.nanmean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap*100

def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return mAP(fix, gt_array)

class EPICTrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.loss_verb = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_verb_total = 0.0
        self.loss_noun = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_noun_total = 0.0
        self.lr = None
        # Current minibatch accuracies (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Number of correctly classified examples.
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.loss_verb.reset()
        self.loss_verb_total = 0.0
        self.loss_noun.reset()
        self.loss_noun_total = 0.0
        self.lr = None
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.mb_verb_top1_acc.reset()
        self.mb_verb_top5_acc.reset()
        self.mb_noun_top1_acc.reset()
        self.mb_noun_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_acc, top5_acc, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_verb_top1_acc.add_value(top1_acc[0])
        self.mb_verb_top5_acc.add_value(top5_acc[0])
        self.mb_noun_top1_acc.add_value(top1_acc[1])
        self.mb_noun_top5_acc.add_value(top5_acc[1])
        self.mb_top1_acc.add_value(top1_acc[2])
        self.mb_top5_acc.add_value(top5_acc[2])
        self.loss_verb.add_value(loss[0])
        self.loss_noun.add_value(loss[1])
        self.loss.add_value(loss[2])
        self.lr = lr
        # Aggregate stats
        self.num_verb_top1_cor += top1_acc[0] * mb_size
        self.num_verb_top5_cor += top5_acc[0] * mb_size
        self.num_noun_top1_cor += top1_acc[1] * mb_size
        self.num_noun_top5_cor += top5_acc[1] * mb_size
        self.num_top1_cor += top1_acc[2] * mb_size
        self.num_top5_cor += top5_acc[2] * mb_size
        self.loss_verb_total += loss[0] * mb_size
        self.loss_noun_total += loss[1] * mb_size
        self.loss_total += loss[2] * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "verb_top1_acc": self.mb_verb_top1_acc.get_win_median(),
            "verb_top5_acc": self.mb_verb_top5_acc.get_win_median(),
            "noun_top1_acc": self.mb_noun_top1_acc.get_win_median(),
            "noun_top5_acc": self.mb_noun_top5_acc.get_win_median(),
            "top1_acc": self.mb_top1_acc.get_win_median(),
            "top5_acc": self.mb_top5_acc.get_win_median(),
            "verb_loss": self.loss_verb.get_win_median(),
            "noun_loss": self.loss_noun.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        verb_top1_acc = self.num_verb_top1_cor / self.num_samples
        verb_top5_acc = self.num_verb_top5_cor / self.num_samples
        noun_top1_acc = self.num_noun_top1_cor / self.num_samples
        noun_top5_acc = self.num_noun_top5_cor / self.num_samples
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        avg_loss_verb = self.loss_verb_total / self.num_samples
        avg_loss_noun = self.loss_noun_total / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "verb_top1_acc": verb_top1_acc,
            "verb_top5_acc": verb_top5_acc,
            "noun_top1_acc": noun_top1_acc,
            "noun_top5_acc": noun_top5_acc,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "verb_loss": avg_loss_verb,
            "noun_loss": avg_loss_noun,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

class EPICValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch accuracies (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Max accuracies (over the full val set).
        self.max_top1_acc = 0.0
        self.max_top5_acc = 0.0
        self.max_verb_top1_acc = 0.0
        self.max_verb_top5_acc = 0.0
        self.max_noun_top1_acc = 0.0
        self.max_noun_top5_acc = 0.0
        # Number of correctly classified examples.
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.mb_verb_top1_acc.reset()
        self.mb_verb_top5_acc.reset()
        self.mb_noun_top1_acc.reset()
        self.mb_noun_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_acc, top5_acc, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            mb_size (int): mini batch size.
        """
        self.mb_verb_top1_acc.add_value(top1_acc[0])
        self.mb_verb_top5_acc.add_value(top5_acc[0])
        self.mb_noun_top1_acc.add_value(top1_acc[1])
        self.mb_noun_top5_acc.add_value(top5_acc[1])
        self.mb_top1_acc.add_value(top1_acc[2])
        self.mb_top5_acc.add_value(top5_acc[2])
        self.num_verb_top1_cor += top1_acc[0] * mb_size
        self.num_verb_top5_cor += top5_acc[0] * mb_size
        self.num_noun_top1_cor += top1_acc[1] * mb_size
        self.num_noun_top5_cor += top5_acc[1] * mb_size
        self.num_top1_cor += top1_acc[2] * mb_size
        self.num_top5_cor += top5_acc[2] * mb_size
        self.num_samples += mb_size
        self.all_preds = []
        self.all_labels = []

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "verb_top1_acc": self.mb_verb_top1_acc.get_win_median(),
            "verb_top5_acc": self.mb_verb_top5_acc.get_win_median(),
            "noun_top1_acc": self.mb_noun_top1_acc.get_win_median(),
            "noun_top5_acc": self.mb_noun_top5_acc.get_win_median(),
            "top1_acc": self.mb_top1_acc.get_win_median(),
            "top5_acc": self.mb_top5_acc.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        verb_top1_acc = self.num_verb_top1_cor / self.num_samples
        verb_top5_acc = self.num_verb_top5_cor / self.num_samples
        noun_top1_acc = self.num_noun_top1_cor / self.num_samples
        noun_top5_acc = self.num_noun_top5_cor / self.num_samples
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        self.max_verb_top1_acc = max(self.max_verb_top1_acc, verb_top1_acc)
        self.max_verb_top5_acc = max(self.max_verb_top5_acc, verb_top5_acc)
        self.max_noun_top1_acc = max(self.max_noun_top1_acc, noun_top1_acc)
        self.max_noun_top5_acc = max(self.max_noun_top5_acc, noun_top5_acc)
        is_best_epoch = top1_acc > self.max_top1_acc
        self.max_top1_acc = max(self.max_top1_acc, top1_acc)
        self.max_top5_acc = max(self.max_top5_acc, top5_acc)
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "verb_top1_acc": verb_top1_acc,
            "verb_top5_acc": verb_top5_acc,
            "noun_top1_acc": noun_top1_acc,
            "noun_top5_acc": noun_top5_acc,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "max_verb_top1_acc": self.max_verb_top1_acc,
            "max_verb_top5_acc": self.max_verb_top5_acc,
            "max_noun_top1_acc": self.max_noun_top1_acc,
            "max_noun_top5_acc": self.max_noun_top5_acc,
            "max_top1_acc": self.max_top1_acc,
            "max_top5_acc": self.max_top5_acc,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

        return is_best_epoch

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds[0])
        self.all_labels.append(labels['verb'])

class MultiLossTrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss_fuse = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_fuse_total = 0.0
        self.loss_vis = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_vis_total = 0.0
        self.loss_box = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_box_total = 0.0
        self.lr = None
        # Current minibatch accuracies (smoothed over a window).
        self.mb_fuse_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_fuse_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_vis_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_vis_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_box_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_box_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Number of correctly classified examples.
        self.num_fuse_top1_cor = 0
        self.num_fuse_top5_cor = 0
        self.num_vis_top1_cor = 0
        self.num_vis_top5_cor = 0
        self.num_box_top1_cor = 0
        self.num_box_top5_cor = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss_fuse.reset()
        self.loss_fuse_total = 0.0
        self.loss_vis.reset()
        self.loss_vis_total = 0.0
        self.loss_box.reset()
        self.loss_box_total = 0.0
        self.lr = None
        self.mb_fuse_top1_acc.reset()
        self.mb_fuse_top5_acc.reset()
        self.mb_vis_top1_acc.reset()
        self.mb_vis_top5_acc.reset()
        self.mb_box_top1_acc.reset()
        self.mb_box_top5_acc.reset()
        self.num_fuse_top1_cor = 0
        self.num_fuse_top5_cor = 0
        self.num_vis_top1_cor = 0
        self.num_vis_top5_cor = 0
        self.num_box_top1_cor = 0
        self.num_box_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_acc, top5_acc, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_fuse_top1_acc.add_value(top1_acc[0])
        self.mb_fuse_top5_acc.add_value(top5_acc[0])
        self.mb_vis_top1_acc.add_value(top1_acc[1])
        self.mb_vis_top5_acc.add_value(top5_acc[1])
        self.mb_box_top1_acc.add_value(top1_acc[2])
        self.mb_box_top5_acc.add_value(top5_acc[2])
        self.loss_fuse.add_value(loss[0])
        self.loss_vis.add_value(loss[1])
        self.loss_box.add_value(loss[2])
        self.lr = lr
        # Aggregate stats
        self.num_fuse_top1_cor += top1_acc[0] * mb_size
        self.num_fuse_top5_cor += top5_acc[0] * mb_size
        self.num_vis_top1_cor += top1_acc[1] * mb_size
        self.num_vis_top5_cor += top5_acc[1] * mb_size
        self.num_box_top1_cor += top1_acc[2] * mb_size
        self.num_box_top5_cor += top5_acc[2] * mb_size
        self.loss_fuse_total += loss[0] * mb_size
        self.loss_vis_total += loss[1] * mb_size
        self.loss_box_total += loss[2] * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "fuse_top1_acc": self.mb_fuse_top1_acc.get_win_avg(),
            "fuse_top5_acc": self.mb_fuse_top5_acc.get_win_avg(),
            "vis_top1_acc": self.mb_vis_top1_acc.get_win_avg(),
            "vis_top5_acc": self.mb_vis_top5_acc.get_win_avg(),
            "box_top1_acc": self.mb_box_top1_acc.get_win_avg(),
            "box_top5_acc": self.mb_box_top5_acc.get_win_avg(),
            "fuse_loss": self.loss_fuse.get_win_avg(),
            "vis_loss": self.loss_vis.get_win_avg(),
            "box_loss": self.loss_box.get_win_avg(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        fuse_top1_acc = self.num_fuse_top1_cor / self.num_samples
        fuse_top5_acc = self.num_fuse_top5_cor / self.num_samples
        vis_top1_acc = self.num_vis_top1_cor / self.num_samples
        vis_top5_acc = self.num_vis_top5_cor / self.num_samples
        box_top1_acc = self.num_box_top1_cor / self.num_samples
        box_top5_acc = self.num_box_top5_cor / self.num_samples
        avg_loss_fuse = self.loss_fuse_total / self.num_samples
        avg_loss_vis = self.loss_vis_total / self.num_samples
        avg_loss_box = self.loss_box_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "fuse_top1_acc": fuse_top1_acc,
            "fuse_top5_acc": fuse_top5_acc,
            "vis_top1_acc": vis_top1_acc,
            "vis_top5_acc": vis_top5_acc,
            "box_top1_acc": box_top1_acc,
            "box_top5_acc": box_top5_acc,
            "fuse_loss": avg_loss_fuse,
            "vis_loss": avg_loss_vis,
            "box_loss": avg_loss_box,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

class MultiLossValMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch accuracies (smoothed over a window).
        self.mb_fuse_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_fuse_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_vis_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_vis_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_box_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_box_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Max accuracies (over the full val set).
        self.max_fuse_top1_acc = 0.0
        self.max_fuse_top5_acc = 0.0
        self.max_vis_top1_acc = 0.0
        self.max_vis_top5_acc = 0.0
        self.max_box_top1_acc = 0.0
        self.max_box_top5_acc = 0.0
        # Number of correctly classified examples.
        self.num_fuse_top1_cor = 0
        self.num_fuse_top5_cor = 0
        self.num_vis_top1_cor = 0
        self.num_vis_top5_cor = 0
        self.num_box_top1_cor = 0
        self.num_box_top5_cor = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_fuse_top1_acc.reset()
        self.mb_fuse_top5_acc.reset()
        self.mb_vis_top1_acc.reset()
        self.mb_vis_top5_acc.reset()
        self.mb_box_top1_acc.reset()
        self.mb_box_top5_acc.reset()
        self.num_fuse_top1_cor = 0
        self.num_fuse_top5_cor = 0
        self.num_vis_top1_cor = 0
        self.num_vis_top5_cor = 0
        self.num_box_top1_cor = 0
        self.num_box_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_acc, top5_acc, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_fuse_top1_acc.add_value(top1_acc[0])
        self.mb_fuse_top5_acc.add_value(top5_acc[0])
        self.mb_vis_top1_acc.add_value(top1_acc[1])
        self.mb_vis_top5_acc.add_value(top5_acc[1])
        self.mb_box_top1_acc.add_value(top1_acc[2])
        self.mb_box_top5_acc.add_value(top5_acc[2])
        # Aggregate stats
        self.num_fuse_top1_cor += top1_acc[0] * mb_size
        self.num_fuse_top5_cor += top5_acc[0] * mb_size
        self.num_vis_top1_cor += top1_acc[1] * mb_size
        self.num_vis_top5_cor += top5_acc[1] * mb_size
        self.num_box_top1_cor += top1_acc[2] * mb_size
        self.num_box_top5_cor += top5_acc[2] * mb_size
        self.num_samples += mb_size
        self.all_preds = []
        self.all_labels = []

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "fuse_top1_acc": self.mb_fuse_top1_acc.get_win_median(),
            "fuse_top5_acc": self.mb_fuse_top5_acc.get_win_median(),
            "vis_top1_acc": self.mb_vis_top1_acc.get_win_median(),
            "vis_top5_acc": self.mb_vis_top5_acc.get_win_median(),
            "box_top1_acc": self.mb_box_top1_acc.get_win_median(),
            "box_top5_acc": self.mb_box_top5_acc.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        fuse_top1_acc = self.num_fuse_top1_cor / self.num_samples
        fuse_top5_acc = self.num_fuse_top5_cor / self.num_samples
        vis_top1_acc = self.num_vis_top1_cor / self.num_samples
        vis_top5_acc = self.num_vis_top5_cor / self.num_samples
        box_top1_acc = self.num_box_top1_cor / self.num_samples
        box_top5_acc = self.num_box_top5_cor / self.num_samples
        self.max_vis_top1_acc = max(self.max_vis_top1_acc, vis_top1_acc)
        self.max_vis_top5_acc = max(self.max_vis_top5_acc, vis_top5_acc)
        self.max_box_top1_acc = max(self.max_box_top1_acc, box_top1_acc)
        self.max_box_top5_acc = max(self.max_box_top5_acc, box_top5_acc)
        is_best_epoch = fuse_top1_acc > self.max_fuse_top1_acc
        self.max_fuse_top1_acc = max(self.max_fuse_top1_acc, fuse_top1_acc)
        self.max_fuse_top5_acc = max(self.max_fuse_top5_acc, fuse_top5_acc)
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "fuse_top1_acc": fuse_top1_acc,
            "fuse_top5_acc": fuse_top5_acc,
            "vis_top1_acc": vis_top1_acc,
            "vis_top5_acc": vis_top5_acc,
            "box_top1_acc": box_top1_acc,
            "box_top5_acc": box_top5_acc,
            "max_fuse_top1_acc": self.max_fuse_top1_acc,
            "max_fuse_top5_acc": self.max_fuse_top5_acc,
            "max_vis_top1_acc": self.max_vis_top1_acc,
            "max_vis_top5_acc": self.max_vis_top5_acc,
            "max_box_top1_acc": self.max_box_top1_acc,
            "max_box_top5_acc": self.max_box_top5_acc,
        }
        logging.log_json_stats(stats)

        return is_best_epoch

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds[0])
        self.all_labels.append(labels)

class TestMeter:
    def __init__(self,
                 num_videos,
                 num_cls,
                 overall_iters,
                 cfg) -> None:
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.overall_iters = overall_iters
        self.video_preds = torch.zeros((num_videos, num_cls))

        self.video_labels = (
            torch.zeros((num_videos)).long()
        )
        self.topk_accs = []
        self.stats = {}

        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)

        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.video_preds.zero_()
        self.video_labels.zero_()
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()

    def update_stats(self, preds, labels, clip_ids):
        top1_acc, top5_acc = metrics.topk_accuracies(
            preds, labels, (1, 5)
        )
        self.mb_top1_acc.add_value(top1_acc.item())
        self.mb_top5_acc.add_value(top5_acc.item())
        
        for ind in range(preds.shape[0]):
            vid_id = clip_ids[ind].item()
            self.video_labels[vid_id] = labels[ind]
            self.video_preds[vid_id] = preds[ind]

    def log_iter_stats(self, cur_iter):
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
            "top1_acc": self.mb_top1_acc.get_win_avg(),
            "top5_acc": self.mb_top5_acc.get_win_avg()
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        self.stats = {"split": "test_final"}
        num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
        topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
        assert len({len(ks), len(topks)}) == 1
        for k, topk in zip(ks, topks):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        logging.log_json_stats(self.stats)
