import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes, # only for testing
    iou_width_height as iou,
    non_max_suppression as nms, # only for testing
    plot_image #only for testing
)
from dataset import YoloDataset

def test():
    anchors = config.ANCHORS

    transform = config.train_transforms

    dataset = YoloDataset(
        config.DATASET+'/train',
        config.IMG_DIR,
        config.LABEL_DIR,
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()