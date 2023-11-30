import torch
import torch.nn as nn
import numpy as np
import cv2
import datetime
from utils.utils import AvgMeter
import torch.nn.functional as F
from net.OrNet import build_model, weights_init
from tqdm import tqdm
from utils.losses import Criterion, Optimizer, Scheduler
from utils.metrics import Evaluation_metrics
from dataloader import gt_to_tensor, get_loader, get_test_augmentation
import os
import time


class Solver(object):
    def __init__(self, train_loader, valid_loader, test_loader, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.lr_decay_epoch = [20]
        self.build_model()
        self.epoch = 0
        self.criterion = Criterion(config)
        self.optimizer = Optimizer(config, self.net)
        self.scheduler = Scheduler(config, self.optimizer)
        self.FG_branch = config.FG_branch
        self.BG_branch = config.BG_branch
        assert self.FG_branch + self.BG_branch

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}M".format(num_params / 1e6))

    # build the network
    def build_model(self):
        if self.config.mode == 'train':
            self.net = build_model(self.config)
            self.print_network(self.net, 'PoolNet Structure')
        else:
            print('Loading pre-trained model from \n%s...' % self.config.load_test)

            if torch.cuda.is_available():
                model = torch.load(self.config.load_test)
                print('epoch:%d' % model['epoch'])
            else:
                model = torch.load(self.config.load_test, map_location='cpu')
                print('epoch:%d' % model['epoch'])
            self.net = model['model']
            self.net.load_state_dict(model['state_dict'])

    def train(self):
        min_mae = 1000
        min_loss = 1000
        early_stopping = 0
        t = time.time()
        best_epoch_mae = 0
        best_epoch_loss = 0
        best_mae = 0

        if torch.cuda.is_available():
            self.net.cuda()
        # Train time
        for epoch in range(1, self.config.epochs + 1):
            self.epoch = epoch
            train_loss, train_mae = self.train_one_epoch()
            val_loss, val_mae = self.valid()

            if self.config.scheduler == 'Reduce':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            with open(os.path.join(self.config.save_folder, 'valid.txt'), 'a+') as f:
                f.write(
                    'Epoch %d: val_loss %f, val_mae %f | %s\n' %
                    (epoch, val_loss, val_mae, str(datetime.datetime.now())))
                f.close()

            # Save models
            if val_loss < min_loss and epoch > 3:
                best_epoch_loss = epoch
                min_loss = val_loss
                torch.save({
                    'epoch': best_epoch_loss,
                    'model': self.net,
                    'state_dict': self.net.state_dict(),
                }, '%s/best_model_loss.pth' % self.config.save_folder)
                print(f'-----------------SAVE:{best_epoch_loss}epoch----------------')

            if val_mae < min_mae and epoch > 3:
                early_stopping = 0
                best_epoch_mae = epoch
                best_mae = val_mae
                min_mae = val_mae
                torch.save({
                    'epoch': best_epoch_mae,
                    'model': self.net,
                    'state_dict': self.net.state_dict(),
                }, '%s/best_model_mae.pth' % self.config.save_folder)
                print(f'-----------------SAVE:{best_epoch_mae}epoch----------------')
            else:
                early_stopping += 1

            if early_stopping == self.config.patience + 5:
                break

        print(f'\nBest Val_mae Epoch:{best_epoch_mae}, Val MAE:{best_mae:.3f} | '
              f'Best Val_loss Epoch:{best_epoch_loss}, Val loss:{min_loss:.3f}'
              f'time: {(time.time() - t) / 3600:.3f}h')
        with open(os.path.join(self.config.save_folder, 'valid.txt'), 'a+') as f:
            f.write('Train Over.\n')
            f.write(f' Best Val MAE Epoch:{best_epoch_mae},   Val MAE:{best_mae:.3f} | '
                    f'Best Val Loss Epoch:{best_epoch_loss}, Val loss:{min_loss:.3f}'
                    f'time: {(time.time() - t) / 3600:.3f}h')
            f.close()

        # Test time
        with open(os.path.join(self.config.save_folder, 'valid.txt'), 'a+') as f:
            f.write("Testing best_model_mae:\n")
            f.close()
        self.net.load_state_dict(torch.load(os.path.join(self.config.save_folder, 'best_model_mae.pth'))['state_dict'])
        datasets = ['DUTS', 'DUT-O', 'HKU-IS', 'ECSSD', 'PASCAL-S']
        for dataset in datasets:
            self.config.dataset = dataset
            test_loss, test_mae, test_maxf, test_avgf, test_s_m = self.test(self.config.save_folder)

            print(
                f'Test Loss:{test_loss:.3f} | MAX_F:{test_maxf:.3f} | AVG_F:{test_avgf:.3f} | MAE:{test_mae:.3f} '
                f'| S_Measure:{test_s_m:.3f}, time: {time.time() - t:.3f}s')
            with open(os.path.join(self.config.save_folder, 'valid.txt'), 'a+') as f:
                f.write(
                    '%s:MAE:%.3f  AVG_F:%.3f  S_Measure:%.3f | %s\n' %
                    (dataset, test_mae, test_avgf, test_s_m, str(datetime.datetime.now())))
                f.close()

        with open(os.path.join(self.config.save_folder, 'valid.txt'), 'a+') as f:
            f.write("Testing best_model_loss:\n")
            f.close()
        self.net.load_state_dict(torch.load(os.path.join(self.config.save_folder, 'best_model_loss.pth'))['state_dict'])
        datasets = ['DUTS', 'DUT-O', 'HKU-IS', 'ECSSD', 'PASCAL-S']
        for dataset in datasets:
            self.config.dataset = dataset
            test_loss, test_mae, test_maxf, test_avgf, test_s_m = self.test(self.config.save_folder)

            print(
                f'Test Loss:{test_loss:.3f} | MAX_F:{test_maxf:.3f} | AVG_F:{test_avgf:.3f} | MAE:{test_mae:.3f} '
                f'| S_Measure:{test_s_m:.3f}, time: {time.time() - t:.3f}s')
            with open(os.path.join(self.config.save_folder, 'valid.txt'), 'a+') as f:
                f.write(
                    '%s:MAE:%.3f  AVG_F:%.3f  S_Measure:%.3f | %s\n' %
                    (dataset, test_mae, test_avgf, test_s_m, str(datetime.datetime.now())))
                f.close()

        end = time.time()
        print(f'Total Process time:{(end - t) / 60:.3f}Minute')

    def train_one_epoch(self):
        self.net.train()
        train_loss = AvgMeter()
        train_mae = AvgMeter()
        for images, masks, edges, b_masks in tqdm(self.train_loader):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            masks = torch.tensor(masks, device=self.device, dtype=torch.float32)
            edges = torch.tensor(edges, device=self.device, dtype=torch.float32)
            b_masks = torch.tensor(b_masks, device=self.device, dtype=torch.float32)
            if (images.size()[2] != masks.size()[2]) or (images.size()[3] != masks.size()[3]):
                print('IMAGE ERROR, PASSING```')
                continue

            outputs_fg, outputs_bg = self.net(images)

            if self.FG_branch:
                loss = self.criterion(outputs_fg[:, 0, :, :].unsqueeze(1), masks)
                if outputs_fg.size(1) > 1:
                    for i in range(1, outputs_fg.size(1)):
                        loss += self.criterion(outputs_fg[:, i, :, :].unsqueeze(1), masks)  # Foreground loss

                if self.BG_branch:
                    loss += self.criterion(outputs_bg[:, 0, :, :].unsqueeze(1), masks)
                    if outputs_bg.size(1) > 1:
                        for i in range(1, outputs_bg.size(1)):
                            loss += self.criterion(outputs_bg[:, i, :, :].unsqueeze(1), masks)  # Foreground loss

            else:
                loss = self.criterion(outputs_bg[:, 0, :, :].unsqueeze(1), masks)
                if outputs_bg.size(1) > 1:
                    for i in range(1, outputs_bg.size(1)):
                        loss += self.criterion(outputs_bg[:, i, :, :].unsqueeze(1), masks)  # Foreground loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.config.clipping)
            self.optimizer.step()

            # Metric
            if self.FG_branch:
                mae = torch.mean(torch.abs(outputs_fg - masks))
                if self.BG_branch:
                    mae += torch.mean(torch.abs(outputs_bg - masks))
                    mae /= 2
            else:
                mae = torch.mean(torch.abs(outputs_bg - masks))

            # log
            train_loss.update(loss.item(), n=images.size()[0])
            train_mae.update(mae.item(), n=images.size()[0])

        print(f'Epoch:[{self.epoch:03d}/{self.config.epochs:03d}]')
        print(f'Train Loss:{train_loss.avg:.3f} | MAE:{train_mae.avg:.3f}')

        return train_loss.avg, train_mae.avg

    def test(self, path):
        te_img_folder = os.path.join(self.config.data_path, self.config.dataset, 'Test/images/')
        te_gt_folder = os.path.join(self.config.data_path, self.config.dataset, 'Test/masks/')
        test_transform = get_test_augmentation(img_size=self.config.img_size)
        test_loader = get_loader(te_img_folder, te_gt_folder, edge_folder=None, phase='test',
                                 batch_size=self.config.batch_size, shuffle=False,
                                 num_workers=self.config.num_workers, transform=test_transform)

        self.net.eval()
        test_loss = AvgMeter()
        test_mae = AvgMeter()
        test_maxf = AvgMeter()
        test_avgf = AvgMeter()
        test_s_m = AvgMeter()

        Eval_tool = Evaluation_metrics(self.config.dataset, self.device)

        with torch.no_grad():
            for k, (images, masks, original_size, image_name) in enumerate(tqdm(test_loader)):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)

                outputs_fg, outputs_bg = self.net(images)

                if len(outputs_fg) > 0:
                    outputs = outputs_fg[:, 0, :, :].unsqueeze(1)
                    if len(outputs_bg) > 0:
                        outputs += outputs_bg[:, 0, :, :].unsqueeze(1)
                        outputs /= 2
                else:
                    outputs = outputs_bg[:, 0, :, :].unsqueeze(1)

                H, W = original_size

                for i in range(images.size()[0]):
                    mask = gt_to_tensor(masks[i])

                    h, w = H[i].item(), W[i].item()

                    output = F.interpolate(outputs[i].unsqueeze(0), size=(h, w), mode='bilinear')

                    loss = self.criterion(output, mask)

                    # Metric
                    mae, max_f, avg_f, s_score = Eval_tool.cal_total_metrics(output, mask)

                    # log
                    test_loss.update(loss.item(), n=1)
                    test_mae.update(mae, n=1)
                    test_maxf.update(max_f, n=1)
                    test_avgf.update(avg_f, n=1)
                    test_s_m.update(s_score, n=1)

                    # Save
                    if self.config.save_map:
                        output = (output.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
                        dir = os.path.join(self.config.test_fold)
                        if not os.path.exists(dir):
                            os.mkdir(dir)
                        dir = os.path.join(dir, self.config.dataset)
                        if not os.path.exists(dir):
                            os.mkdir(dir)
                        cv2.imwrite(os.path.join(dir, image_name[i] + '.png'), output)

            test_loss = test_loss.avg
            test_mae = test_mae.avg
            test_maxf = test_maxf.avg
            test_avgf = test_avgf.avg
            test_s_m = test_s_m.avg

        return test_loss, test_mae, test_maxf, test_avgf, test_s_m

    def valid(self):
        self.net.eval()
        val_loss = AvgMeter()
        val_mae = AvgMeter()

        with torch.no_grad():
            for images, masks, edges, b_masks in tqdm(self.valid_loader):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                masks = torch.tensor(masks, device=self.device, dtype=torch.float32)
                edges = torch.tensor(edges, device=self.device, dtype=torch.float32)

                outputs_fg, outputs_bg = self.net(images)

                if self.FG_branch:
                    loss = self.criterion(outputs_fg[:, 0, :, :].unsqueeze(1), masks)
                    if outputs_fg.size(1) > 1:
                        for i in range(1, outputs_fg.size(1)):
                            loss += self.criterion(outputs_fg[:, i, :, :].unsqueeze(1), masks)  # Foreground loss

                    if self.BG_branch:
                        loss += self.criterion(outputs_bg[:, 0, :, :].unsqueeze(1), masks)
                        if outputs_bg.size(1) > 1:
                            for i in range(1, outputs_bg.size(1)):
                                loss += self.criterion(outputs_bg[:, i, :, :].unsqueeze(1), masks)  # Foreground loss

                else:
                    loss = self.criterion(outputs_bg[:, 0, :, :].unsqueeze(1), masks)
                    if outputs_bg.size(1) > 1:
                        for i in range(1, outputs_bg.size(1)):
                            loss += self.criterion(outputs_bg[:, i, :, :].unsqueeze(1), masks)  # Foreground loss

                # Metric
                if self.FG_branch:
                    mae = torch.mean(torch.abs(outputs_fg - masks))
                    if self.BG_branch:
                        mae += torch.mean(torch.abs(outputs_bg - masks))
                        mae /= 2
                else:
                    mae = torch.mean(torch.abs(outputs_bg - masks))

                # log
                val_loss.update(loss.item(), n=images.size()[0])
                val_mae.update(mae, n=images.size()[0])

        print(f'Valid Loss:{val_loss.avg:.3f} | MAE:{val_mae.avg:.3f}')
        return val_loss.avg, val_mae.avg
