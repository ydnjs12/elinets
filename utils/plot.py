import cv2
import webcolors
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


color_list = standard_to_bgr(STANDARD_COLORS)


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


class ConfusionMatrix:
    def __init__(self, num_classes=6):
        """
        Args:
            num_classes: The total number of lane classes.
        """
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int32)  # Square matrix for classification

    def update(self, preds, labels):
        """
        Update the confusion matrix given a batch of predictions and labels.

        Args:
            preds (Array[N, 2]): Predicted lane numbers conf, class.
            labels (Array[N]): Ground truth lane numbers (from 0 to num_classes-1).
        """
        gt_class = labels.int()
        pred_class = preds[:, 1].int()

        self.matrix[gt_class, pred_class] += 1


    def tp_fp_fn(self):
        """
        Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) for each class.
        Returns:
            tp: True Positives
            fp: False Positives
            fn: False Negatives
        """
        tp = np.diag(self.matrix)  # True Positives: diagonal elements
        fp = self.matrix.sum(axis=0) - tp  # False Positives: sum of the column - TP
        fn = self.matrix.sum(axis=1) - tp  # False Negatives: sum of the row - TP

        return tp, fp, fn

    def plot(self, class_names=None, normalize=False, save_dir=None):
        """
        Plot the confusion matrix using Seaborn heatmap.

        Args:
            class_names (List[str], optional): List of class names to display on axes.
            normalize (bool): Whether to normalize the confusion matrix by row (true class).
            save_dir (str, optional): Directory to save the plot. If None, the plot will just be shown.
        """
        if normalize:
            matrix = self.matrix.astype('float') / (self.matrix.sum(axis=1)[:, np.newaxis] + 1e-6)
            matrix[matrix < 0.005] = np.nan
        else:
            matrix = self.matrix


        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.0 if self.num_classes < 50 else 0.8)
        sns.heatmap(matrix, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', square=True,
                    xticklabels=class_names + ['background FP'],
                    yticklabels=class_names + ['background FP']).set_facecolor((1, 1, 1))

        plt.xlabel('Predicted Lane')
        plt.ylabel('True Lane')
        plt.title('Confusion Matrix')

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        # plt.show()
        plt.close()

    def print(self):
        """Print the confusion matrix in plain text."""
        print("Confusion Matrix:")
        print(self.matrix)
