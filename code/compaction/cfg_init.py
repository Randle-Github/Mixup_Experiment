import argparse
from utils import str2bool

def main_args():
    parser = argparse.ArgumentParser(description='PyTorch Smth-Else')
    parser.add_argument(
        '--dataset_name', choices=['sth_else', 'charades'], default='sth_else')
    parser.add_argument(
        '--dataset_root', default='dataset')
    parser.add_argument('--model', default='coord')
    args, _ = parser.parse_known_args()

    parser.add_argument('--root_frames', default='%s/%s/frames' %
                        (args.dataset_root, args.dataset_name), type=str, help='path to the folder with frames')
    parser.add_argument('--json_data_train', default='dataset_splits/%s/compositional/train.json' %
                        (args.dataset_name), type=str, help='path to the json file with train video meta data')
    parser.add_argument('--json_data_val', default='dataset_splits/%s/compositional/validation.json' %
                        (args.dataset_name), type=str, help='path to the json file with validation video meta data')
    parser.add_argument('--json_file_labels', default='dataset_splits/%s/compositional/labels.json' %
                        (args.dataset_name), type=str, help='path to the json file with ground truth labels')
    parser.add_argument('--tracked_boxes', default='dataset/%s/bounding_box_annotations.pkl' %
                        (args.dataset_name), type=str, help='choose tracked boxes')

    parser.add_argument('--img_feature_dim', default=512, type=int, metavar='N',
                        help='intermediate feature dimension for image-based features')
    parser.add_argument('--coord_feature_dim', default=512, type=int, metavar='N',
                        help='intermediate feature dimension for coord-based features')
    parser.add_argument('--edge_feature_dim', default=512, type=int, metavar='N',
                        help='feature dim for edge between subject and object')
    parser.add_argument('--msg_feature_dim', default=512, type=int, metavar='N',
                        help='feature dim for message')
    parser.add_argument('--node_feature_dim', default=512, type=int, metavar='N',
                        help='feature dim for node')
    parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                        metavar='W', help='gradient norm clipping (default: 5)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--size', default=224, type=int, metavar='N',
                        help='primary image input size')
    parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', '-b', default=72, type=int,
                        metavar='N', help='mini-batch size (default: 72)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', default=[10, 15, 20], type=float, nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--log_freq', '-l', default=10, type=int,
                        metavar='N', help='frequency to write in tensorboard (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--num_classes', default=174, type=int,
                        help='num of class in the model')
    parser.add_argument('--num_boxes', default=4, type=int,
                        help='num of boxes for each image')
    parser.add_argument('--num_frames', default=8, type=int,
                        help='num of frames for the model')
    parser.add_argument('--dataset', default='smth_smth',
                        help='which dataset to train')
    parser.add_argument('--logdir', default='./logs',
                        help='folder to output tensorboard logs')
    parser.add_argument('--logname', default='exp',
                        help='name of the experiment for checkpoints and logs')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--fine_tune', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--shot', default=0)
    parser.add_argument('--restore_i3d')
    parser.add_argument('--restore_custom')
    parser.add_argument('--if_augment', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--psuedo', type=str2bool, const=True, default=True,
                        nargs='?', help='是否使用组合增强训练模型,')
    parser.add_argument('--local_rank', type=int, default=-1)

    args, _ = parser.parse_known_args()
    return args

def model_args():
    # 这部分参数用于设置模型的具体设计
    parser = argparse.ArgumentParser()
    parser.add_argument('--i3d', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--hidden_feature_dim', type=int, default=512)
    parser.add_argument('--learning_method', type=str, choices=[
        'vanilla', 'pairwise', 'msgpassing', 'psuedo', 'vgae'])
    parser.add_argument('--vgae', type=str2bool, const=True, default=True,
                        nargs='?', help='是否使用 VGAE 得到组合,')
    parser.add_argument('--fusion', type=str, default='cat', help='verb 和 noun feature 融合的方式')
    parser.add_argument('--coc_matrix', type=str, choices=['train', 'val', 'complete', 'random', 'none', 'gae'], default='gae', help='使用哪种共现矩阵')
    parser.add_argument('--glob', type=str2bool, const=True, default=False, nargs='?' ,help='是否使用 global 的 RGB 信息')
    parser.add_argument('--backbone', type=str, default='i3d', help='使用的骨架网络')
    parser.add_argument('--slowfast_head', type=str, default='basic', help='slowfast 的 head 选取')
    args, _ = parser.parse_known_args()
    return args

def data_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_order', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--vis_info', type=str2bool, nargs='?',
                        const=True, default=False)
    args, _ = parser.parse_known_args()
    return args

def solver_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_lr', default=0.01, type=float,
                        metavar='LR', help='base learning rate')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_policy', default='consine', type=str,
                        help='学习率衰减的策略')
    parser.add_argument('--warmup_epochs', default=0, type=float,
                        help='在前 n 个 epoch 中 warmup')
    parser.add_argument('--warmup_start_lr', default=0.001, type=float,
                        help='warmup 的起始 lr')
    parser.add_argument('--cosine_end_lr', default=0, type=float,
                        help='cosine decay 最后的 lr')
    parser.add_argument('--cosine_after_warmup', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')

    args, _ = parser.parse_known_args()
    return args
