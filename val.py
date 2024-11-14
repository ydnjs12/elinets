import torch
import numpy as np
import argparse
from tqdm.autonotebook import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
from utils import smp_metrics
from utils.plot import ConfusionMatrix
from utils.utils import postprocess, ap_per_class, fitness, \
    save_checkpoint, DataLoaderX, boolean_string, Params
from backbone import HybridNetsBackbone
from elinets.dataset import BddDataset
from torchvision import transforms
import torch.nn.functional as F
from elinets.model import ModelWithLoss
from utils.constants import *
import cv2
import torchmetrics

iou_metric = torchmetrics.JaccardIndex(num_classes=6, task="multiclass").to("cuda")
f1_metric = torchmetrics.F1Score(num_classes=6, task="multiclass").to("cuda")

@torch.no_grad()
def val(model, val_generator, params, opt, seg_mode, is_training, **kwargs):
    model.eval()
    
    saved_path = f'checkpoints/{opt.project}/'
    log_path = f'checkpoints/{opt.project}/tensorboard/'
    save_dir = 'plots'
    os.makedirs(saved_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = kwargs.get('optimizer', None)
    scaler = kwargs.get('scaler', None)
    writer = kwargs.get('writer', None)
    epoch = kwargs.get('epoch', 0)
    step = kwargs.get('step', 0)
    best_fitness = kwargs.get('best_fitness', 0)
    best_loss = kwargs.get('best_loss', 0)
    best_epoch = kwargs.get('best_epoch', 0)

    loss_classification_ls = []
    loss_segmentation_ls = []
    iou_ls = [[] for _ in range(len(params.seg_list)+1)]
    acc_ls = [[] for _ in range(len(params.seg_list)+1)]
    iou_ls_mean = [[] for _ in range(len(params.seg_list)+1)]
    acc_ls_mean = [[] for _ in range(len(params.seg_list)+1)]
    
    classification_metrics = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0}
    segmentation_metrics = {"miou": 0.0, "accuracy": 0.0}

    total_correct = 0
    total_samples = 0

    stats = []
    names = {i: v for i, v in enumerate(params.label_list)}
    seen = 0
    confusion_matrix = ConfusionMatrix(num_classes=len(params.label_list))
    s_seg = ' ' * (15 + 11 * 6)
    s = ('%-15s' + '%-11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mIoU', 'mAcc')
    for i in range(len(params.seg_list)):
        s_seg += '%-33s' % params.seg_list[i]
        s += ('%-11s' * 3) % ('mIoU', 'IoU', 'Acc')
    p, r, f1, mp, mr = 0.0, 0.0, 0.0, 0.0, 0.0
    
    iou_metric.reset()
    f1_metric.reset()

    val_loader = tqdm(val_generator, ascii=True)
    for iter, data in enumerate(val_loader):
        imgs = data['img'].cuda()
        total_lane = data['totalLane'].cuda()
        ego_lane = data['egoLane'].cuda()
        seg_annot = data['segmentation'].cuda()

        with torch.amp.autocast(device_type=device, enabled=opt.amp):
            cls_loss, seg_loss, classification, segmentation = model(imgs, ego_lane, seg_annot, label_list=params.label_list)
        
        cls_loss = cls_loss.mean()
        seg_loss = seg_loss.mean()
        
        loss = cls_loss + seg_loss
        if loss == 0 or not torch.isfinite(loss):
            continue

        loss_classification_ls.append(cls_loss.item())
        loss_segmentation_ls.append(seg_loss.item())

        # Segmentation Metrics
        if seg_mode == MULTICLASS_MODE:
            segmentation = segmentation.log_softmax(dim=1).exp()
            _, segmentation = torch.max(segmentation, 1)  # (bs, C, H, W) -> (bs, H, W)
        else:
            segmentation = F.logsigmoid(segmentation).exp()

        tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(segmentation, seg_annot, mode=seg_mode,
                                                                threshold=0.5 if seg_mode != MULTICLASS_MODE else None,
                                                                num_classes=len(params.seg_list)+1 if seg_mode == MULTICLASS_MODE else None)
        iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
        acc = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
        
        for i in range(len(params.seg_list)+1):
            iou_ls[i].append(iou.T[i].detach().cpu().numpy())
            acc_ls[i].append(acc.T[i].detach().cpu().numpy())
        
        # Classification Metrics
        out = postprocess(imgs.detach(), classification.detach(), total_lane)
        correct_cnt = 0

        for i in range(ego_lane.size(0)):
            seen += 1
            labels = ego_lane[i].cuda()

            ou = out[i]

            pred = np.column_stack([ou['scores'], ou['class_ids']])
            pred = torch.from_numpy(pred).cuda()

            if len(pred) == 0:
                if labels:
                    stats.append((torch.zeros(0, dtype=torch.bool), torch.Tensor(), torch.Tensor(), labels))
                continue

            if labels:
                correct = (pred[:, 1].clone().detach() == labels).unsqueeze(0)
                correct_cnt = correct_cnt + 1 if correct == True else correct_cnt
                confusion_matrix.update(pred[:, 1], labels)
            else:
                correct = torch.zeros(pred.shape[0], dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 0].cpu(), pred[:, 1].cpu(), [labels.tolist()]))

        total_correct += correct_cnt
        total_samples += ego_lane.size(0)

        iou_metric.update(torch.tensor([item['class_ids'][0] for item in out]).cuda(), ego_lane)
        f1_metric.update(torch.tensor([item['class_ids'][0] for item in out]).cuda(), ego_lane)
        
    validation_iou = iou_metric.compute()
    validation_f1 = f1_metric.compute()
    print(f"Validation IoU: {validation_iou:.4f}, F1 Score: {validation_f1:.4f}")

    cls_loss = np.mean(loss_classification_ls)
    seg_loss = np.mean(loss_segmentation_ls)
    
    cls_weight = 1
    seg_weight = 1

    loss = cls_weight * cls_loss + seg_weight * seg_loss

    for i in range(len(params.seg_list)+1):
        iou_ls[i] = np.concatenate(iou_ls[i])
        acc_ls[i] = np.concatenate(acc_ls[i])
    iou_score = np.mean(iou_ls)
    acc_score = np.mean(acc_ls)
    
    for i in range(len(params.seg_list)+1):
        iou_ls_mean[i] = np.mean(iou_ls[i])
        acc_ls_mean[i] = np.mean(acc_ls[i])

    miou_ls = []
    for i in range(len(params.seg_list)):
        if seg_mode == BINARY_MODE:
            # typically this runs once with i == 0
            miou_ls.append(np.mean(iou_ls[i]))
        else:
            miou_ls.append(np.mean((iou_ls[0] + iou_ls[i+1]) / 2))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    metrics_accruacy = total_correct/total_samples
    # Compute metrics
    if len(stats) and stats[0].any():
        p, r, f1, _, _ = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
        classification_metrics["precision"] = p.mean().item()
        classification_metrics["recall"] = r.mean().item()
        classification_metrics["f1_score"] = f1.mean().item()
        classification_metrics["accuracy"] = (p.mean() + r.mean()) / 2

        # Per-class classification accuracy
        class_precision = p
        class_recall = r
        class_f1_score = f1
        class_accuracy = (class_precision + class_recall) / 2

        # Print per-class classification metrics
        print("Per-Class Classification Results:")
        for idx in range(min(len(class_precision), len(names))):
            class_name = names[idx]
            print(f"Class '{class_name}' - Precision: {class_precision[idx]:.5f}, "
                f"Recall: {class_recall[idx]:.5f}, F1 Score: {class_f1_score[idx]:.5f}, "
                f"Accuracy: {class_accuracy[idx]:.5f}")

        nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    print('Val. Epoch: {}/{}. Classification loss: {:1.5f}. Segmentation loss: {:1.5f}. Total loss: {:1.5f}'.format(
            epoch, opt.num_epochs if is_training else 0, cls_loss, seg_loss, loss))
    if is_training:
        writer.add_scalars('Loss/Total', {'val': loss}, step)
        writer.add_scalars('Loss/Classfication', {'val': cls_loss}, step)
        writer.add_scalars('Loss/Segmentation', {'val': seg_loss}, step)

        writer.add_scalar('Metrics/Classification Precision', classification_metrics['precision'], step)
        writer.add_scalar('Metrics/Classification Recall', classification_metrics['recall'], step)
        writer.add_scalar('Metrics/Classification F1 Score', classification_metrics['f1_score'], step)
        writer.add_scalar('Metrics/Classification Accuracy', classification_metrics['accuracy'], step)
        writer.add_scalar('Metrics/Segmentation mIoU', iou_score, step)
        writer.add_scalar('Metrics/Segmentation Accuracy', acc_score, step)

        writer.add_scalar("Metrics/Validation IoU", validation_iou, step)
        writer.add_scalar("Metrics/Validation F1 Score", validation_f1, step)

    print(s_seg)
    print(s)
    pf = ('%-15s' + '%-11i' * 2 + '%-11.3g' * 4) % (
        'all', seen, nt.sum(), classification_metrics["precision"], classification_metrics["recall"], 
        iou_score, acc_score)
    for i in range(len(params.seg_list)):
        tmp = i+1 if seg_mode != BINARY_MODE else i
        pf += ('%-11.3g' * 3) % (miou_ls[i], iou_ls_mean[tmp], acc_ls_mean[tmp])
    print(pf)

    # Plots
    confusion_matrix.plot(save_dir=save_dir, normalize=True, class_names=list(names.values()))
    confusion_matrix.tp_fp_fn()

    results = (classification_metrics["precision"], classification_metrics["recall"], iou_score, acc_score, loss)
    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, iou, acc, loss]

    print(f"Validation Results:\n"
      f"Classification Loss: {cls_loss:.5f}\n"
      f"Segmentation Loss: {seg_loss:.5f}\n"
      f"Total Loss: {loss:.5f}\n"
      f"Segmentation IoU Score: {iou_score:.5f}\n"
      f"Segmentation Accuracy Score: {acc_score:.5f}\n"
      f"Classification Precision: {classification_metrics['precision']:.5f}\n"
      f"Classification Recall: {classification_metrics['recall']:.5f}\n"
      f"Classification F1 Score: {classification_metrics['f1_score']:.5f}\n"
      f"Classification Accuracy: {classification_metrics['accuracy']:.5f}\n"
      f"metrics_accruacy: {metrics_accruacy:.5f}")
    
    print("Per-Class Segmentation Results:")
    for idx, class_name in enumerate(params.seg_list + ["background"]):
        print(f"Class '{class_name}' - IoU: {iou_ls_mean[idx]:.5f}, Accuracy: {acc_ls_mean[idx]:.5f}")
    
    # if calculating map, save by best fitness
    if is_training and fi > best_fitness:
        best_fitness = fi
        ckpt = {'epoch': epoch,
                'step': step,
                'best_fitness': best_fitness,
                'model': model.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict()}
        print("Saving checkpoint with best fitness", fi[0])
        save_checkpoint(ckpt, saved_path, f'elinets-d{opt.compound_coef}_{epoch}_{step}_best.pth')
        
    # if not calculating map, save by best loss
    elif is_training and loss + opt.es_min_delta < best_loss:
        best_loss = loss
        best_epoch = epoch

        save_checkpoint(model, saved_path, f'elinets-d{opt.compound_coef}_{epoch}_{step}_best.pth')

    # Early stopping
    if is_training and epoch - best_epoch > opt.es_patience > 0:
        print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
        exit(0)

    model.train()
    return (best_fitness, best_loss, best_epoch) if is_training else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    ap.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                        'https://github.com/rwightman/pytorch-image-models')
    ap.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficients of efficientnet backbone')
    ap.add_argument('-w', '--weights', type=str, default='weights/hybridnets.pth', help='/path/to/weights')
    ap.add_argument('-n', '--num_workers', type=int, default=12, help='Num_workers of dataloader')
    ap.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    args = ap.parse_args()

    weights_path = f'weights/elinets-d{args.compound_coef}.pth' if args.weights is None else args.weights

    params = Params(f'projects/{args.project}.yml')
    seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE

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
        seg_mode=seg_mode
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    model = HybridNetsBackbone(compound_coef=args.compound_coef, num_classes=len(params.label_list),
                               seg_classes=len(params.seg_list), backbone_name=args.backbone,
                               seg_mode=seg_mode)
    
    try:
        model.load_state_dict(torch.load(weights_path))
    except:
        model.load_state_dict(torch.load(weights_path)['model'])
    model = ModelWithLoss(model, debug=False)
    model.requires_grad_(False)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model = model.to("cuda")
    else :
        model = model.to("cpu")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    val(model, val_generator, params, args, seg_mode, is_training=False)
