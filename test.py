import time
import torch
import json
from torch.backends import cudnn
from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from utils.utils import letterbox, postprocess, restricted_float, boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
from torchvision import transforms
import argparse
from utils.constants import *
from torch.nn import functional as F
from tqdm.autonotebook import tqdm


parser = argparse.ArgumentParser('Multi-task Learning for Ego-lane Inference')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                        'https://github.com/rwightman/pytorch-image-models')
parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
parser.add_argument('--source', type=str, default='demo', help='The demo image folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result to output folder")
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
parser.add_argument('--show_seg', type=boolean_string, default=False, help="Output segmentation result exclusively")
parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
parser.add_argument('--speed_test', type=boolean_string, default=False, help='Measure inference latency')
args = parser.parse_args()

params = Params(f'projects/{args.project}.yml')
use_cuda = torch.cuda.is_available()
use_float16 = args.float16
cudnn.fastest = True
cudnn.benchmark = True

weight = torch.load(args.load_weights, map_location='cuda' if use_cuda else 'cpu')
#new_weight = OrderedDict((k[6:], v) for k, v in weight['model'].items())
weight_last_layer_seg = weight['segmentation_head.0.weight']
if weight_last_layer_seg.size(0) == 1:
    seg_mode = BINARY_MODE
else:
    if params.seg_multilabel:
        seg_mode = MULTILABEL_MODE
    else:
        seg_mode = MULTICLASS_MODE

color_list_seg = {}
color_list_seg['road'] = list((255,0,0))
color_list_seg['lane'] = list((0,255,0))
# for seg_class in params.seg_list:
#     color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))

output = args.output
output = output[:-1] if output.endswith("/") else output
os.makedirs(output, exist_ok=True)

label_list = params.label_list
seg_list = params.seg_list

test_path = f'{params.labelroot}/drivable_test_custom.json'
with open(test_path, 'r') as f:
    test_file = json.load(f)
    file = [i for i in test_file]

ori_imgs = []
ori_totallanes = []
ori_gt = []
for data in tqdm(file, ascii=True):
    image_path = f'{params.dataroot}/test/{data['name']}'
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    total_lane = int(data['info'][0]['totalLane'])
    ground_truth = int(data['info'][0]['egoLane'])
    ori_imgs.append(img)
    ori_totallanes.append(total_lane)
    ori_gt.append(ground_truth)
print(f"FOUND {len(ori_imgs)} IMAGES")

resized_shape = params.model['image_size']
if isinstance(resized_shape, list): resized_shape = max(resized_shape)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=params.mean, std=params.std
    ),
])

input_imgs = []
shapes = []
det_only_imgs = []

for ori_img in ori_imgs:
    h0, w0 = ori_img.shape[:2]  # orig hw
    r = resized_shape / max(h0, w0)  # resize image to img_size
    input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
    h, w = input_img.shape[:2]

    (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True, scaleup=False)

    input_imgs.append(input_img)
    # cv2.imwrite('input.jpg', input_img * 255)
    shapes.append(((h0, w0), ((h / h0, w / w0), pad)))  # for COCO mAP rescaling

if use_cuda:
    x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
else:
    x = torch.stack([transform(fi) for fi in input_imgs], 0)

x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)

print("DETECTED SEGMENTATION MODE FROM WEIGHT AND PROJECT FILE:", seg_mode)
model = HybridNetsBackbone(compound_coef=args.compound_coef, num_classes=len(label_list), seg_classes=len(seg_list),
                            backbone_name=args.backbone, seg_mode=seg_mode)
model.load_state_dict(weight)
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
    if use_float16:
        model = model.half()

with torch.no_grad(): 
    features, classification, segmentation = model(x)

    # in case of MULTILABEL_MODE, each segmentation class gets their own inference image
    seg_mask_list = []
    # (B, C, W, H) -> (B, W, H)
    if seg_mode == BINARY_MODE:
        seg_mask = torch.where(segmentation >= 0, 1, 0).squeeze_(1)
        seg_mask_list.append(seg_mask)
    elif seg_mode == MULTICLASS_MODE:
        _, seg_mask = torch.max(segmentation, 1)
        seg_mask_list.append(seg_mask)
    else:
        seg_mask_list = [torch.where(torch.sigmoid(segmentation)[:, i, ...] >= 0.5, 1, 0) for i in range(segmentation.size(1))]
        # but remove background class from the list
        seg_mask_list.pop(0)

    # (B, W, H) -> (W, H)
    for i in range(segmentation.size(0)):
        for seg_class_index, seg_mask in enumerate(seg_mask_list):
            seg_mask_ = seg_mask[i].squeeze().cpu().numpy()
            pad_h = int(shapes[i][1][1][1])
            pad_w = int(shapes[i][1][1][0])
            seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
            seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)
            color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
            for index, seg_class in enumerate(params.seg_list):
                color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
            color_seg = color_seg[..., ::-1]  # RGB -> BGR
            # cv2.imwrite('seg_only_{}.jpg'.format(i), color_seg)

            color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background

            seg_img = ori_imgs[i].copy() if seg_mode == MULTILABEL_MODE else ori_imgs[i]  # do not work on original images if MULTILABEL_MODE
            seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
            seg_img = seg_img.astype(np.uint8)
            seg_filename = f'{output}/{i}_{params.seg_list[seg_class_index]}_seg.jpg' if seg_mode == MULTILABEL_MODE else f'{output}/{i}_seg.jpg'
            if args.show_seg or seg_mode == MULTILABEL_MODE:
                cv2.imwrite(seg_filename, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))

    out = postprocess(x, classification, total_lane)

    for i in range(len(ori_imgs)):
        label = str(out[i]['class_ids'])
        score = float(out[i]['scores'])

        if args.imwrite:
            total_lane_text = f'total_lane: {total_lane[i]} '
            ego_lane_text = f'ego_lane: {label} '
            score_text = f'score: {score:.2f}'
            gt_txt = f'GT: {int(ground_truth[i])} '

            cv2.putText(ori_imgs[i], total_lane_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(ori_imgs[i], ego_lane_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(ori_imgs[i], score_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(ori_imgs[i], gt_txt, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imwrite(f'{output}/{i+1}.jpg', cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))
            print(f'{i}.jpg : , ego_lane_index : {label} ')
            
if not args.speed_test:
    exit(0)
print('running speed test...')
with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring 1 image for 10 times...')
    x = x[0, ...]
    x.unsqueeze_(0)
    t1 = time.time()
    for _ in range(10):
        _, classification, segmentation = model(x)

        out = postprocess(x, classification, total_lane)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    # uncomment this if you want a extreme fps test
    print('test2: model inferring only')
    print('inferring images for batch_size 32 for 10 times...')
    t1 = time.time()
    x = torch.cat([x] * 32, 0)
    for _ in range(10):
        _, classification, segmentation = model(x)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
