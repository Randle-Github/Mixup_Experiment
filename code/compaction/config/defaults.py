# 这个文件里面将存放所有需要配置的参数，并为他们设置一些默认值
# 这个默认值可以被 yaml 或者命令行参数所替代

from fvcore.common.config import CfgNode

# Config definition ###########################################################
_C = CfgNode()

# Training options ############################################################
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# batch size 是针对每个 node (machine) 而言
_C.TRAIN.BATCH_SIZE = 16

# Dataset.
_C.TRAIN.DATASET = "sth_else"

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# 设置为 true, model 将返回多个 branch 的预测结果
_C.TRAIN.MULTI_LOSS = False

# 自己的 mixup, 在特征层面
_C.TRAIN.MIXUP = False

# mixup 方式的选择
_C.TRAIN.MIXER = 'random'

#
_C.TRAIN.FS_FINETUNE = False

#
_C.TRAIN.CUSTOM_SAMPLER = False

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = False

_C.TEST.CHECKPOINT_FILE_PATH = ""

_C.TEST.CHECKPOINT_TYPE = "pytorch"

_C.TEST.SAVE_RESULTS_PATH = ""

# -----------------------------------------------------------------------------
# ResNet options
# 针对 slowfast 的默认设置，所以列表的每个元素都有两个值
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = True

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3, 3], [4, 4], [6, 6], [3, 3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1, 1], [2, 2], [2, 2], [1, 1]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1, 1], [1, 1], [1, 1], [1, 1]]

# -----------------------------------------------------------------------------
# Nonlocal options
# 实际没有用到 nonlocal, 这里都是一些默认设置，并不会生效
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[], []], [[], []], [[], []], [[], []]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1, 1], [1, 1], [1, 1], [1, 1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"

# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Model options
# 这里面没有提供选 head 吗？
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
# 我们这里当作是骨架网络的选取
_C.MODEL.ARCH = "slowfast"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

#
_C.MODEL.HEAD = 'basichead'

_C.MODEL.HEAD_RESIDUAL = False

_C.MODEL.HEAD_CONCAT = False

_C.MODEL.HEAD_STAGE = -1

# 如果设置为 true, 那么除 head 部分，其他网络参数均 fix
_C.MODEL.FINETUNE = False

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 174

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Model architectures that has single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["i3d"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# 是否使用 cls token
_C.MODEL.USE_CLS = True

# 指定一个 pretrain 的 slowfast. 单独提取特征
_C.MODEL.INTER_TRUNK = ''
_C.MODEL.OBJ_TRUNK = ''

# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 4

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 7

# ho 推理的参数 ##############################################################
_C.REASON = CfgNode()

# 第一项表示 cross attention 的层数，第二项表示 self attention 的层数
_C.REASON.VIS_DEPTHS = [1, 1]

_C.REASON.BOX_DEPTHS = [1, 1]

# 单独先加载 box model 的参数
_C.REASON.BOX_PRETRAIN = ''

# 仅对 box 的 HoReasonNet 使用 cls token
_C.REASON.BOX_USE_CLS = False

# 是否使用 hand/obj identity.
_C.REASON.IDENTITY = False

# 加 spatial position embedding 的方式，有 box, grid, none 三种
_C.REASON.SPE = 'grid'


# ---------------------------------------------------------------------------- #
# VIT options
# ---------------------------------------------------------------------------- #
_C.VIT = CfgNode()

# Patch-size spatial to tokenize input
_C.VIT.PATCH_SIZE = 16

# Patch-size temporal to tokenize input
_C.VIT.PATCH_SIZE_TEMP = 2

# Number of input channels
_C.VIT.CHANNELS = 3

# Embedding dimension
_C.VIT.EMBED_DIM = 768

# Depth of transformer: number of layers
_C.VIT.DEPTH = 12

# number of attention heads
_C.VIT.NUM_HEADS = 12

# expansion ratio for MLP
_C.VIT.MLP_RATIO = 4

# add bias to QKV projection layer
_C.VIT.QKV_BIAS = True

# video input
_C.VIT.VIDEO_INPUT = True

# temporal resolution i.e. number of frames
_C.VIT.TEMPORAL_RESOLUTION = 8

# use MLP classification head
_C.VIT.USE_MLP = False

# Dropout rate for
_C.VIT.DROP = 0.0

# Stochastic drop rate
_C.VIT.DROP_PATH = 0.0

# Dropout rate for MLP head
_C.VIT.HEAD_DROPOUT = 0.0

# Dropout rate for positional embeddings
_C.VIT.POS_DROPOUT = 0.0

# Dropout rate 
_C.VIT.ATTN_DROPOUT = 0.0

# Activation for head
_C.VIT.HEAD_ACT = "tanh"

# Use IM pretrained weights
_C.VIT.IM_PRETRAINED = True

# Pretrained weights type
_C.VIT.PRETRAINED_WEIGHTS = "vit_1k"

_C.VIT.USE_POS = 'none'

# 在每层 attention 的时候都加上 position encoding
_C.VIT.POS_EACH_LAYER = False

# 加 position embedding 的方式。‘add’ 'cat'.
_C.VIT.ATTACH_POS = 'add'

# Type of position embedding
_C.VIT.POS_EMBED = "separate"

# Self-Attention layer
_C.VIT.ATTN_LAYER = "trajectory"


# -----------------------------------------------------------------------------
# Data options
# 原本有很多参数，但我只保留了会用到的一些
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "~/FSL_Datasets/sth-sth-v2"

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 32

# 
_C.DATA.NUM_BOXES = 5

# List of input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

#
_C.DATA.MEAN = [0.485, 0.456, 0.406]

_C.DATA.STD = [0.229, 0.224, 0.225]

# If True, use repeated aug
_C.DATA.USE_REPEATED_AUG = False

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# 在 sthsth 的数据集里面，有 ground truth 和 detected 两种 bbox. 他们的命名有差别，默认用 ground truth.
_C.DATA.BOUNDING_BOX_NAME = 'bounding_box_smthsmth' # detected_bounding_boxes

# 选择 sthelse 的数据集划分方式。compositional 和 shuffled.
_C.DATA.STHELSE_SPLIT = 'compositional'
# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #

_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
# TODO: 这个参数不是很明白
_C.BN.NUM_SYNC_DEVICES = 1

# 用自己的 sync bn
_C.BN.CUSTOM_SYNC = False

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 1.0

# 针对 box 网络的 lr.
_C.SOLVER.BOX_LR = 0.0001

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# Use Mixed Precision Training
_C.SOLVER.USE_MIXED_PRECISION = False

# If > 0.0, use label smoothing
_C.SOLVER.SMOOTHING = 0.0

# Clip Grad
_C.SOLVER.CLIP_GRAD = None

# 使用自己的 optimer
_C.SOLVER.CUSTOM_OPTIM = False

# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = False

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 20

# If True, log the model info.
_C.LOG_MODEL_INFO = False

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# Distributed backend.
_C.USE_SBATCH = True

#
_C.DEBUG = False

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""
# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
# This file must be provided to enable plotting confusion matrix
# by a subset or parent categories.
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

# Path to a json file for categories -> classes mapping
# in the format {"parent_class": ["child_class1", "child_class2",...], ...}.
_C.TENSORBOARD.CATEGORIES_PATH = ""

# Config for confusion matrices visualization.
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
# Visualize confusion matrix.
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
# Figure size of the confusion matrices plotted.
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
# Path to a subset of categories to visualize.
# File contains class names separated by newline characters.
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

# Config for histogram visualization.
_C.TENSORBOARD.HISTOGRAM = CfgNode()
# Visualize histograms.
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
# Path to a subset of classes to plot histograms.
# Class names must be separated by newline characters.
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
# Visualize top-k most predicted classes on histograms for each
# chosen true label.
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
# Figure size of the histograms plotted.
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]

# Config for layers' weights and activations visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS = CfgNode()

# If False, skip model visualization.
_C.TENSORBOARD.MODEL_VIS.ENABLE = False

# If False, skip visualizing model weights.
_C.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = False

# If False, skip visualizing model activations.
_C.TENSORBOARD.MODEL_VIS.ACTIVATIONS = False

# If False, skip visualizing input videos.
_C.TENSORBOARD.MODEL_VIS.INPUT_VIDEO = False


# List of strings containing data about layer names and their indexing to
# visualize weights and activations for. The indexing is meant for
# choosing a subset of activations outputed by a layer for visualization.
# If indexing is not specified, visualize all activations outputed by the layer.
# For each string, layer name and indexing is separated by whitespaces.
# e.g.: [layer1 1,2;1,2, layer2, layer3 150,151;3,4]; this means for each array `arr`
# along the batch dimension in `layer1`, we take arr[[1, 2], [1, 2]]
_C.TENSORBOARD.MODEL_VIS.LAYER_LIST = []
# Top-k predictions to plot on videos
_C.TENSORBOARD.MODEL_VIS.TOPK_PREDS = 1
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.COLORMAP = "Pastel2"
# Config for visualization video inputs with Grad-CAM.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM = CfgNode()
# Whether to run visualization using Grad-CAM technique.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
# CNN layers to use for Grad-CAM. The number of layers must be equal to
# number of pathway(s).
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = []
# If True, visualize Grad-CAM using true labels for each instances.
# If False, use the highest predicted class.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP = "viridis"

# Config for visualization for wrong prediction visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.WRONG_PRED_VIS = CfgNode()
_C.TENSORBOARD.WRONG_PRED_VIS.ENABLE = False
# Folder tag to origanize model eval videos under.
_C.TENSORBOARD.WRONG_PRED_VIS.TAG = "Incorrectly classified videos."
# Subset of labels to visualize. Only wrong predictions with true labels
# within this subset is visualized.
_C.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH = ""

# ---------------------------------------------------------------------------- #
# Mixup options
# ---------------------------------------------------------------------------- #

_C.MIXUP = CfgNode()

# mixup alpha, mixup enabled if > 0. (default: 0.8)
_C.MIXUP.MIXUP_ALPHA = 0.0

# cutmix alpha, cutmix enabled if > 0. (default: 1.0)
_C.MIXUP.CUTMIX_ALPHA = 0.0

# cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
_C.MIXUP.CUTMIX_MINMAX = None

# Probability of performing mixup or cutmix when either/both is enabled
_C.MIXUP.MIXUP_PROB = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled
_C.MIXUP.MIXUP_SWITCH_PROB = 0.5

# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.MIXUP.MIXUP_MODE = "batch"

# 
_C.MIXUP.ENABLE = False

def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    # assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    # assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    # assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
