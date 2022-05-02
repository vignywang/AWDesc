#
# Created  on 2020/2/23
#
import numpy as np
import torch
import torch.nn.functional as f


class Matcher(object):

    def __init__(self, dtype='float'):
        if dtype == 'float':
            self.compute_desp_dist = self._compute_desp_dist
        elif dtype == 'binary':
            self.compute_desp_dist = self._compute_desp_dist_binary
        else:
            assert False

    def __call__(self, point_0, desp_0, point_1, desp_1):
        dist_0_1 = self.compute_desp_dist(desp_0, desp_1)  # [n,m]
        dist_1_0 = dist_0_1.transpose((1, 0))  # [m,n]
        nearest_idx_0_1 = np.argmin(dist_0_1, axis=1)  # [n]
        nearest_idx_1_0 = np.argmin(dist_1_0, axis=1)  # [m]
        matched_src = []
        matched_tgt = []
        for i, idx_0_1 in enumerate(nearest_idx_0_1):
            if i == nearest_idx_1_0[idx_0_1]:
                matched_src.append(point_0[i])
                matched_tgt.append(point_1[idx_0_1])
        if len(matched_src) <= 4:
            print("There exist too little matches")
            # assert False
            return None
        if len(matched_src) != 0:
            matched_src = np.stack(matched_src, axis=0)
            matched_tgt = np.stack(matched_tgt, axis=0)
        return matched_src, matched_tgt

    @staticmethod
    def _compute_desp_dist(desp_0, desp_1):
        # desp_0:[n,256], desp_1:[m,256]
        square_norm_0 = (np.linalg.norm(desp_0, axis=1, keepdims=True)) ** 2  # [n,1]
        square_norm_1 = (np.linalg.norm(desp_1, axis=1, keepdims=True).transpose((1, 0))) ** 2  # [1,m]
        xty = np.matmul(desp_0, desp_1.transpose((1, 0)))  # [n,m]
        dist = np.sqrt((square_norm_0 + square_norm_1 - 2 * xty + 1e-4))
        return dist

    @staticmethod
    def _compute_desp_dist_binary(desp_0, desp_1):
        # desp_0:[n,256], desp_1[m,256]
        dist_0_1 = np.logical_xor(desp_0[:, np.newaxis, :], desp_1[np.newaxis, :, :]).sum(axis=2)
        return dist_0_1


def spatial_nms(prob, kernel_size=9):
    """
    利用max_pooling对预测的特征点的概率图进行非极大值抑制
    Args:
        prob: shape为[h,w]的概率图
        kernel_size: 对每个点进行非极大值抑制时的窗口大小

    Returns:
        经非极大值抑制后的概率图
    """
    padding = int(kernel_size//2)
    pooled = f.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=padding)
    prob = torch.where(torch.eq(prob, pooled), prob, torch.zeros_like(prob))
    return prob


def convert_cv2pt(cv_point):
    point_list = []
    for i, cv_pt in enumerate(cv_point):
        pt = np.array((cv_pt.pt[1], cv_pt.pt[0]))  # y,x的顺序
        point_list.append(pt)
    point = np.stack(point_list, axis=0)
    return point


def model_size(model):
    ''' Computes the number of parameters of the model
    '''
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


def torch_set_gpu(gpus):
    import os
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu>=0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'],os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True # speed-up cudnn
        torch.backends.cudnn.fastest = True # even more speed-up?
        print( 'Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'] )

    else:
        print( 'Launching on CPU' )

    return cuda




