import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from Dataloaders.DATApath import Path
from Dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.AmRTSNet import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
import torch
from thop import profile

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="PyTorch AmRTSNet Training")
    parser.add_argument('--dataset', type=str, default='rrhtdata')
    parser.add_argument('--backbone', type=str, default='mobilenet', help='backbone name')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--loss-type', type=str, default='mix', choices=['ce', 'focal', 'Dice', 'mix'],
                        help='loss type')
    parser.add_argument('--batch-size', type=int, default=11, metavar='N', help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=21, metavar='N', help='testing batch size')
    parser.add_argument('--out-stride', type=int, default=8, help='output stride')
    parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
    parser.add_argument('--sync-bn', type=bool, default=None, help='use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='freeze bn parameters')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epoch')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True, help='balanced weights')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'],
                        help='lr scheduler')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=False, help='use nesterov')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--checkname', type=str, default=None, help='checkpoint name')
    parser.add_argument('--ft', action='store_true', default=False, help='finetune')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    if args.sync_bn is None:
        args.sync_bn = args.cuda and len(args.gpu_ids) > 1
    if args.epochs is None:
        epoches = {'rrhtdata': 100}
        args.epochs = epoches.get(args.dataset.lower(), args.epochs)
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    if args.lr is None:
        lrs = {'rrhtdata': 0.1}
        args.lr = lrs.get(args.dataset.lower(), args.lr) / (4 * len(args.gpu_ids)) * args.batch_size
    if args.checkname is None:
        args.checkname = 'AmRTSNet-' + str(args.backbone)

    print(args)
    torch.manual_seed(args.seed)

    # ────────────────────────────────────────
    temp_model = AmRTSNet(num_classes=2,         # num_classes = 2
                         backbone=args.backbone,
                         output_stride=args.out_stride,
                         sync_bn=False,
                         freeze_bn=False).cpu()
    # 生成一个 dummy 输入：Batch=1, Channels=4, H=crop-size, W=crop-size
    dummy_input = torch.randn(1, 4, args.crop_size, args.crop_size)
    flops, params = profile(temp_model, inputs=(dummy_input,), verbose=False)
    print(f"→ Model parameters: {params/1e6:.2f} M")
    print(f"→ Model FLOPs:      {flops/1e9:.2f} G")
    del temp_model, dummy_input
    # ────────────────────────────────────────
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()
    elapsed = time.time() - start_time
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f'Training Completed in: {int(hrs)}h {int(mins)}m {int(secs)}s')
    # ────────────────────────────────────────
class Trainer(object):
    def __init__(self, args):
        self.args = args
        # ────────────────────────────────────────
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # ────────────────────────────────────────
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # ────────────────────────────────────────
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        # ────────────────────────────────────────
        # Define network
        model = AmRTSNet(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        # ────────────────────────────────────────
        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        # ────────────────────────────────────────
        # Define Criterion
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        # ────────────────────────────────────────
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))
        # ────────────────────────────────────────
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        # ────────────────────────────────────────
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        # ────────────────────────────────────────
        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            torch.backends.cudnn.enabled = False
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description(f'Train loss: {train_loss / (i + 1):.3f}')
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print(f'[Epoch: {epoch}, numImages: {i * self.args.batch_size + image.data.shape[0]}]')
        print(f'Loss: {train_loss:.3f}')

        if self.args.no_val:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            losses = test_loss / (i + 1)
            tbar.set_description(f'Test loss: {losses:.3f}')
            pred = output.data.cpu().numpy()
            target_np = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target_np, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print(f'[Epoch: {epoch}, numImages: {i * self.args.batch_size + image.data.shape[0]}]')
        print(f"Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}, fwIoU: {FWIoU}")
        print(f'Loss: {test_loss:.3f}')

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

if __name__ == '__main__':
    main()
