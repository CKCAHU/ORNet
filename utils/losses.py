import torch
import torch.nn.functional as F


def Optimizer(args, model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # SGD
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return optimizer


def Scheduler(args, optimizer):
    if args.scheduler == 'Reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.patience)
    else:  # Step
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)
    return scheduler


def Criterion(config):
    if config.loss_f == 'BCE':
        criterion = torch.nn.BCELoss()
    elif config.loss_f == 'BDL':  # BDL
        criterion = boundary_diffusion_loss
    else:
        criterion = adaptive_pixel_intensity_loss
    return criterion


def boundary_diffusion_loss(pred, mask):
    m1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    m2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    m3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    p1 = torch.abs(F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1) - pred)
    p2 = torch.abs(F.avg_pool2d(pred, kernel_size=15, stride=1, padding=7) - pred)
    p3 = torch.abs(F.avg_pool2d(pred, kernel_size=31, stride=1, padding=15) - pred)

    m = (m1 + m2 + m3) / 3
    p = (p1 + p2 + p3) / 3

    bce = F.binary_cross_entropy(pred, mask, reduce=None)
    bce += F.binary_cross_entropy(p, m, reduce=None)

    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)

    inter = (p * m).sum(dim=(2, 3))
    union = (p + m).sum(dim=(2, 3))
    iou += 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)
    mae += F.l1_loss(p, m, reduce=None)

    return 0.5 * 0.7 * (bce + iou + mae).mean()


def adaptive_pixel_intensity_loss(pred, mask):
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    omega = 1 + 0.5 * (w1 + w2 + w3) * mask

    bce = F.binary_cross_entropy(pred, mask, reduce=None)
    abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

    inter = ((pred * mask) * omega).sum(dim=(2, 3))
    union = ((pred + mask) * omega).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)
    amae = (omega * mae).sum(dim=(2, 3)) / (omega - 1).sum(dim=(2, 3))

    return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean()
