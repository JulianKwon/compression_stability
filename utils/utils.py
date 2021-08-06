import os
import pickle
import shutil

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


def accuracy(y_data, p_data, s_data):
    """Computes the precision@k for the specified values of k"""

    f1 = f1_score(y_data, p_data, average='micro')
    acc = accuracy_score(y_data, p_data)
    auc = roc_auc_score(y_data, s_data, multi_class='ovo')
    pre = precision_score(y_data, p_data, average='micro')
    rec = recall_score(y_data, p_data, average='micro')

    return {
        'f1': f1,
        'accuracy': acc,
        'auc_score': auc,
        'precision': pre,
        'recall': rec
    }


def reshape_resulting_array(arr):
    if type(arr) is not np.ndarray:
        arr = np.array(arr)
    return arr.reshape((-1, arr.shape[-1]))


def get_flat_fts(in_size, fts, device):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.to(device)
    f = fts(torch.autograd.Variable(dummy_input))
    print('conv_out_size: {}'.format(f.size()))
    return int(np.prod(f.size()[1:]))


def save_checkpoint(state, name, is_best=False, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % name + 'model_best.pth.tar')


def save_data(state, directory, filename='inference_result.pkl'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    with open(filename, 'wb') as f:
        pickle.dump(state, f)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
