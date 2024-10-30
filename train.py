import argparse
import datetime
import os
import traceback

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm

from val import val
from backbone import HybridNetsBackbone
from utils.utils import get_last_weights, init_weights, boolean_string, \
    save_checkpoint, DataLoaderX, Params
from elinets.dataset import BddDataset
from elinets.model import ModelWithLoss
from utils.constants import *


def get_args():
    parser = argparse.ArgumentParser('Multi-task Learning for Ego-lane Inference')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Num_workers of dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Number of images per batch among all devices') #12
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='Select optimizer for training, '
                                                                   'suggest using \'adamw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which '
                             'training will be stopped. Set to 0 to disable this technique')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='Whether to load weights from a checkpoint, set None to initialize,'
                             'set \'last\' to load last checkpoint')
    parser.add_argument('--cal_map', type=boolean_string, default=True,
                        help='Calculate mAP in validation')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                        help='Confidence threshold in NMS')
    parser.add_argument('--iou_thres', type=float, default=0.6,
                        help='IoU threshold in NMS')
    parser.add_argument('--amp', type=boolean_string, default=False,
                        help='Automatic Mixed Precision training')

    args = parser.parse_args()
    return args

def train(opt):
    torch.backends.cudnn.benchmark = True
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
    params = Params(f'projects/{opt.project}.yml')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(42)
    # else:
    #     torch.manual_seed(42)

    # ====== Model Initialization ======
    seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE

    model = HybridNetsBackbone(num_classes=len(params.label_list), compound_coef=opt.compound_coef,
                               seg_classes=len(params.seg_list), backbone_name=opt.backbone,
                               seg_mode=seg_mode)
    
    # wrap the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model)

    model = model.to(memory_format=torch.channels_last)

    if use_cuda:
        model = model.to(device)
    else :
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scaler = torch.amp.GradScaler(device=device, enabled=opt.amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    training_generator, val_generator = initDataLoader(params, seg_mode=seg_mode)

    # load last weights
    ckpt = {}
    if opt.load_weights:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(saved_path)

        try:
            ckpt = torch.load(weights_path)
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print('[Warning] this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
    else:
        print('[Info] initializing weights...')
        init_weights(model)

    print('[Info] Successfully!!!')

    # Timestamp for identifying training sessions
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'{log_path}/{time_str}/')

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    last_step = ckpt['step'] if opt.load_weights is not None and ckpt.get('step', None) else 0
    best_fitness = ckpt['best_fitness'] if opt.load_weights is not None and ckpt.get('best_fitness', None) else 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    try:
        for epoch in range(opt.num_epochs):
            epoch_loss = []
            progress_bar = tqdm(training_generator, ascii=True)

            for iter, data in enumerate(progress_bar):
                try:
                    imgs = data['img']
                    total_lane = data['totalLane']
                    ego_lane = data['egoLane']
                    seg_annot = data['segmentation']

                    if torch.cuda.device_count() > 0:
                        imgs = imgs.to(device=device, memory_format=torch.channels_last)
                        ego_lane = ego_lane.cuda()
                        seg_annot = seg_annot.cuda()

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=str(device), enabled=opt.amp):
                        cls_loss, seg_loss, classification, segmentation = model(imgs, ego_lane, seg_annot,
                                                                                label_list=params.label_list)
                        cls_loss = cls_loss.mean()
                        seg_loss = seg_loss.mean()

                        loss = cls_loss + seg_loss
                        
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    scaler.scale(loss).backward()

                    # Don't have to clip grad norm, since our gradients didn't explode anywhere in the training phases
                    # This worsens the metrics
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Seg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            seg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                    writer.add_scalars('Segmentation_loss', {'train': seg_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, saved_path, f'elinets-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                best_fitness, best_loss, best_epoch = val(model, val_generator, params, opt, seg_mode, is_training=True,
                                                          optimizer=optimizer, scaler=scaler, writer=writer, epoch=epoch, step=step, 
                                                          best_fitness=best_fitness, best_loss=best_loss, best_epoch=best_epoch)
    except KeyboardInterrupt:
        save_checkpoint(model, saved_path, f'elinets-d{opt.compound_coef}_{epoch}_{step}.pth')
    finally:
        writer.close()

def initDataLoader(params, seg_mode):
    train_dataset = BddDataset(
        params=params,
        is_train=True,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
    )

    training_generator = DataLoaderX(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    valid_dataset = BddDataset(
        params=params,
        is_train=False,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    return training_generator, val_generator


if __name__ == '__main__':
    opt = get_args()
    saved_path = f'checkpoints/{opt.project}/'
    log_path = f'checkpoints/{opt.project}/tensorboard/'
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(saved_path, exist_ok=True)

    train(opt)
