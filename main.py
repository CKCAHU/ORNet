import os
import pprint
import random
import torch
import numpy as np
from dataloader import get_train_augmentation, get_test_augmentation, get_loader
from solver import Solver
from config import getConfig
import shutil
import warnings
warnings.filterwarnings('ignore')


import sys

stack = []
stack.append('apple')
stack.append('icecream')
stack.append('watermelon')
stack.append('chips')
stack.append('hotdogs')
stack.append('hotpot')
while(stack):
    stack.pop()
    print(stack)


def main(cfg):
    print('<---- Training Params ---->')
    pprint.pprint(cfg)

    # Random Seed
    seed = cfg.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cfg.mode == 'train':
        run = 1
        while os.path.exists("%s/run-%d" % (cfg.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (cfg.save_folder, run))
        cfg.save_folder = "%s/run-%d" % (cfg.save_folder, run)

        # BackUp
        source_path = os.path.join('net')
        target_path = os.path.join(cfg.save_folder, 'backups', source_path)
        shutil.copytree(source_path, target_path)
        source_path = os.path.join('utils')
        target_path = os.path.join(cfg.save_folder, 'backups', source_path)
        shutil.copytree(source_path, target_path)
        source_path = os.path.join('config.py')
        target_path = os.path.join(cfg.save_folder, 'backups', source_path)
        shutil.copyfile(source_path, target_path)
        source_path = os.path.join('solver.py')
        target_path = os.path.join(cfg.save_folder, 'backups', source_path)
        shutil.copyfile(source_path, target_path)
        source_path = os.path.join('main.py')
        target_path = os.path.join(cfg.save_folder, 'backups', source_path)
        shutil.copyfile(source_path, target_path)
        source_path = os.path.join('main_s.py')
        target_path = os.path.join(cfg.save_folder, 'backups', source_path)
        shutil.copyfile(source_path, target_path)

        # Ready
        tr_img_folder = os.path.join(cfg.data_path, cfg.dataset, 'Train/images/')
        tr_gt_folder = os.path.join(cfg.data_path, cfg.dataset, 'Train/masks/')
        tr_edge_folder = os.path.join(cfg.data_path, cfg.dataset, 'Train/edges/')

        train_transform = get_train_augmentation(img_size=cfg.img_size, ver=cfg.aug_ver)
        test_transform = get_test_augmentation(img_size=cfg.img_size)

        train_loader = get_loader(tr_img_folder, tr_gt_folder, tr_edge_folder, phase='train',
                                  batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  transform=train_transform, seed=cfg.seed)
        val_loader = get_loader(tr_img_folder, tr_gt_folder, tr_edge_folder, phase='val',
                                batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                transform=test_transform, seed=cfg.seed)

        train = Solver(train_loader, val_loader, None, cfg)
        train.train()
        if os.path.exists('job.out'):
            source_path = os.path.join('job.out')
            target_path = os.path.join(cfg.save_folder, 'backups', source_path)
            shutil.copyfile(source_path, target_path)
    elif cfg.mode == 'test':
        datasets = ['DUTS', 'DUT-O', 'HKU-IS', 'ECSSD', 'PASCAL-S']

        for dataset in datasets:
            cfg.dataset = dataset

            test_transform = get_test_augmentation(img_size=cfg.img_size)
            te_img_folder = os.path.join(cfg.data_path, cfg.dataset, 'Test/images/')
            te_gt_folder = os.path.join(cfg.data_path, cfg.dataset, 'Test/masks/')

            test_loader = get_loader(te_img_folder, te_gt_folder, edge_folder=None, phase=cfg.mode,
                                     batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers, transform=test_transform
                                     )

            if not os.path.exists(cfg.test_fold):
                os.mkdir(cfg.test_fold)
            test = Solver(None, None, test_loader, cfg)
            test_loss, test_mae, test_maxf, test_avgf, test_s_m = test.test(cfg.load_test)

            print(f'Test Loss:{test_loss:.3f}\n'
                  f'M:{test_mae:.3f}\nFA:{test_avgf:.3f}\nS:{test_s_m:.3f}')
    else:
        raise IOError("illegal input!")


if __name__ == '__main__':
    config = getConfig()
    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)
    main(config)
