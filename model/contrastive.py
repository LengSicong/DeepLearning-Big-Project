import torch
import math
import torch.nn.functional as F

def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y



def get_positive_expectation(p_samples, measure = 'JSD', average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure='JSD', average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
        
    if average:
        return Eq.mean()
    else:
        return Eq

def video_query_loss(video, query,v_len, s, e, gpu_idx, measure='JSD'):
    '''
    Args:
        g: Global features
        g1: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_clips = video.shape[0] #int(v_len)
    num_query = query.shape[0]

    pos_mask = torch.zeros(num_clips,num_query).cuda(int(gpu_idx))
    label_len = e - s + 1
    for i in range(e-s):
        pos_mask[s+i] = 1.
    pos_mask[e] = 1

    #pos_mask = torch.eye(num_clips).cuda()
    neg_mask = 1 - pos_mask

    neg_mask[int(v_len):] = 0.

    res = torch.mm(video, query.t())
    #res = torch.mm(g_enc, g_enc1.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
    E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
    if label_len == int(v_len):
        E_neg = 0
    else:
        E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
        E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

    return E_neg - E_pos

def video_video_loss(video, v_len, s, e, gpu_idx, measure='JSD'):
    '''
    Args:
        g: Global features
        g1: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_clips = video.shape[0] #int(v_len)

    pos_mask = torch.zeros(num_clips,1).cuda(int(gpu_idx))
    label_len = e - s + 1
    for i in range(e-s):
        pos_mask[s+i] = 1.
    pos_mask[e] = 1.

    neg_mask = 1 - pos_mask

    neg_mask[int(v_len):] = 0.

    res_s = torch.mm(video, video[s].view(1,-1).t())
    res_e = torch.mm(video, video[e].view(1,-1).t())

    E_s_pos = get_positive_expectation(res_s * pos_mask, measure, average=False)
    E_s_pos = (E_s_pos * pos_mask).sum() / pos_mask.sum()
    E_e_pos = get_positive_expectation(res_e * pos_mask, measure, average=False)
    E_e_pos = (E_e_pos * pos_mask).sum() / pos_mask.sum()
    if label_len == int(v_len):
        E_s_neg = 0
        E_e_neg = 0
    else:
        E_s_neg = get_negative_expectation(res_s * neg_mask, measure, average=False)
        E_s_neg = (E_s_neg * neg_mask).sum() / neg_mask.sum()
        E_e_neg = get_negative_expectation(res_e * neg_mask, measure, average=False)
        E_e_neg = (E_e_neg * neg_mask).sum() / neg_mask.sum()

    return E_s_neg + E_e_neg - E_s_pos - E_e_pos