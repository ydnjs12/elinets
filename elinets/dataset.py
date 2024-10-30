import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset
from utils.utils import letterbox
from tqdm.autonotebook import tqdm
import json
from collections import OrderedDict
from utils.constants import *


class BddDataset(Dataset):
    def __init__(self, params, is_train, inputsize=[640, 384], transform=None, seg_mode=MULTICLASS_MODE):
        """
        initial all the characteristic

        Inputs:
        -params: configuration parameters
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.is_train = is_train
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()

        img_root = params.dataset['dataroot']
        label_root = params.dataset['labelroot']
        seg_root = params.dataset['segroot']
        self.label_list = params.label_list
        self.seg_list = params.seg_list
        
        indicator = params.dataset['train_set'] if is_train else params.dataset['test_set']
        self.img_root = f"{img_root}/{indicator}"
        self.label_root = f"{label_root}/drivable_{indicator}_custom_ft.json"
        self.seg_root = [f"{root}/{indicator}" for root in seg_root]

        self.shapes = np.array(params.dataset['org_img_size'])
        self.dataset = params.dataset
        self.seg_mode = seg_mode
        self.db = self._get_db()

    def _get_db(self):
        """
        Read label_file, Set image and segment path
        Read label_data, Extract information
        => Save in DB
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes

        try:
            with open(self.label_root, 'r') as file:
                label_file = json.load(file)
                labels = [i for i in label_file]
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            raise

        for label in tqdm(labels, ascii=True):
            image_path = Path(self.img_root) / label['name']

            seg_path = {}
            for i in range(len(self.seg_list)):
                seg_path[self.seg_list[i]] = Path(self.seg_root[i]) / (label['name'].replace(".jpg", ".png"))

            data = label['info'][0]
            if data['egoLane'] != "" and (int(data['totalLane']) if data['totalLane'] != "" else 10) <= len(self.label_list):
                rec = {
                    'image': image_path,
                    'totalLane': int(data['totalLane']),
                    'egoLane': int(data['egoLane']),
                }
            else:
                continue

            # Since seg_path is a dynamic dict
            rec = {**rec, **seg_path}

            gt_db.append(rec)

        print('database build finish')

        return gt_db


    def evaluate(self, params, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError

    def __len__(self, ):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def load_image(self, index):
        '''
        upload image corresponding index and preprocess
        '''
        data = self.db[index]
        total_lane = data["totalLane"]
        ego_lane = data["egoLane"]
        img = cv2.imread(str(data["image"]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_label = OrderedDict()
        for seg_class in self.seg_list:
            seg_label[seg_class] = cv2.imread(str(data[seg_class]), 0)

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_sizeWW
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            for seg_class in self.seg_list:
                seg_label[seg_class] = cv2.resize(seg_label[seg_class], (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        for seg_class in seg_label:
            _, seg_label[seg_class] = cv2.threshold(seg_label[seg_class], 0, 255, cv2.THRESH_BINARY)
    
        return img, total_lane, ego_lane, seg_label, (h0, w0), (h,w), None

    def __getitem__(self, idx):
        """
        Return : corresponding data item idx,
        letterbox() : image resizing,
        Segmentation mode : segmentation label processing
        """
        img, total_lane, ego_lane, seg_label, (h0, w0), (h, w), path = self.load_image(idx)

        (img, seg_label), ratio, pad = letterbox((img, seg_label), (self.inputsize[1], self.inputsize[0]), auto=False,
                                                             scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling  
        total_lane = np.array([total_lane])
        ego_lane = np.array([ego_lane])

        img = np.ascontiguousarray(img)
        
        if self.seg_mode == BINARY_MODE:
            for seg_class in seg_label:
                # technically, the for-loop only goes once
                segmentation = self.Tensor(seg_label[seg_class])

        elif self.seg_mode == MULTICLASS_MODE:
            # special treatment for lane-line of bdd100k for our dataset
            # since we increase lane-line from 2 to 8 pixels, we must take care of the overlap to other segmentation classes
            # e.g.: a pixel belongs to both road and lane-line, then we must prefer lane, or metrics would be wrong
            if 'line' in seg_label:
                for seg_class in seg_label:
                    if seg_class != 'line': seg_label[seg_class] -= seg_label['line']

            segmentation = np.zeros(img.shape[:2], dtype=np.uint8)
            segmentation = self.Tensor(segmentation)
            segmentation.squeeze_(0)
            for seg_index, seg_class in enumerate(seg_label.values()):
                segmentation[seg_class == 255] = seg_index + 1

            # background = 0, road = 1, line = 2
            # [0, 0, 0, 0]
            # [2, 1, 1, 2]
            # [2, 1, 1, 2]
            # [1, 1, 1, 1]

        else:  # multi-label
            union = np.zeros(img.shape[:2], dtype=np.uint8)
            for seg_class in seg_label:
                union |= seg_label[seg_class]
            background = 255 - union

            for seg_class in seg_label:
                seg_label[seg_class] = self.Tensor(seg_label[seg_class])
            background = self.Tensor(background)
            segmentation = torch.cat([background, *seg_label.values()], dim=0)

            # [C, H, W]
            # background [1, 1, 1, 1] road [0, 0, 0, 0]   line [0, 0, 0, 0]
            #            [0, 0, 0, 0]      [0, 1, 1, 0]        [1, 0, 0, 1]
            #            [0, 0, 0, 0]      [0, 1, 1, 0]        [1, 0, 0, 1]
            #            [0, 0, 0, 0]      [1, 1, 1, 1]        [1, 0, 0, 1]

        img = self.transform(img)

        return img, path, shapes, torch.from_numpy(total_lane), torch.from_numpy(ego_lane), segmentation.long()

    @staticmethod
    def collate_fn(batch):
        img, paths, shapes, total_lane, ego_lane, segmentation = zip(*batch)

        return {'img': torch.stack(img, 0), 
                'totalLane': torch.tensor(total_lane), 
                'egoLane': torch.tensor(ego_lane), 
                'segmentation': torch.stack(segmentation, 0),
                'filenames': None, 
                'shapes': shapes
            }
