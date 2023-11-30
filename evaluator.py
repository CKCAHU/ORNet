import numpy as np
import torch


def MAE(pred, gt):
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    mae = np.mean(np.abs(pred - gt))

    return mae


def Fmeasure(pred, gt):
    beta2 = 0.3
    p_avg = torch.zeros(255)
    r_avg = torch.zeros(255)

    p, r = _eval_pr(pred, gt, 255)
    p_avg += p
    r_avg += r
    f_score = (1 + beta2) * p * r / (beta2 * p + r + 1e-20)
    f_score[f_score != f_score] = 0  # for Nan
    adp_f = _eval_adp_f_measure(pred, gt)

    return f_score.max().item(), f_score.mean().item(), adp_f.item()


def Emeasure(pred, gt):
    scores = _eval_e(pred, gt, 255)
    adp_e = _eval_adp_e(pred, gt)
    return scores.max().item(), scores.mean().item(), adp_e.item()


def Smeasure(pred, gt, alpha=0.7):
    # alpha = 0.7; cited from the F-360iSOD
    gt[gt >= 0.5] = 1
    gt[gt < 0.5] = 0
    y = gt.mean()
    if y == 0:
        x = pred.mean()
        Q = 1.0 - x
    elif y == 1:
        x = pred.mean()
        Q = x
    else:
        # gt[gt>=0.5] = 1
        # gt[gt<0.5] = 0
        Q = alpha * _S_object(pred, gt) + (1 - alpha) * _S_region(pred, gt)
        if Q.item() < 0:
            Q = torch.FloatTensor([0.0])
    avg_q = Q.item()
    if np.isnan(avg_q):
        raise  # error

    return avg_q


def _eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num), torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall


def _eval_adp_f_measure(y_pred, y):
    beta2 = 0.3
    thr = y_pred.mean() * 2
    if thr > 1:
        thr = 1
    y_temp = (y_pred >= thr).float()
    tp = (y_temp * y).sum()
    prec, recall = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    adp_f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall + 1e-20)
    if torch.isnan(adp_f_score):
        adp_f_score = 0.0
    return adp_f_score


def _eval_e(y_pred, y, num):
    score = torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        if torch.mean(y) == 0.0:  # the ground-truth is totally black
            y_pred_th = torch.mul(y_pred_th, -1)
            enhanced = torch.add(y_pred_th, 1)
        elif torch.mean(y) == 1.0:  # the ground-truth is totally white
            enhanced = y_pred_th
        else:  # normal cases
            fm = y_pred_th - y_pred_th.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4

        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)

    return score


def _eval_adp_e(y_pred, y):
    th = y_pred.mean() * 2
    y_pred_th = (y_pred >= th).float()
    if torch.mean(y) == 0.0:  # the ground-truth is totally black
        y_pred_th = torch.mul(y_pred_th, -1)
        enhanced = torch.add(y_pred_th, 1)
    elif torch.mean(y) == 1.0:  # the ground-truth is totally white
        enhanced = y_pred_th
    else:  # normal cases
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
    return torch.sum(enhanced) / (y.numel() - 1 + 1e-20)


def _S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg

    return Q


def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    if torch.isnan(score):
        raise
    return score


def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    return Q


def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = torch.eye(1) * round(cols / 2)
        Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        i = torch.from_numpy(np.arange(0, cols)).float()
        j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)

    return X.long(), Y.long()


def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3

    return LT, RT, LB, RB, w1, w2, w3, w4


def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]

    return LT, RT, LB, RB


def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0

    return Q
