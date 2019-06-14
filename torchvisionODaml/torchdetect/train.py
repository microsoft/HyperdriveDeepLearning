
import sys 
import os
import argparse
sys.path.append('./cocoapi/PythonAPI/')

from PIL import Image
import torch
# import torch.utils.data
import xml.etree.ElementTree as ET

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead

import transforms as T
from engine import train_one_epoch, evaluate
import utils

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'JPEGImages'))))
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'JPEGImages', self.imgs[idx])
        xml_path = os.path.join(self.root, 'Annotations', '{}.xml'.format(self.imgs[idx].strip('.jpg')))
        img = Image.open(img_path).convert("RGB")
        
        # parse XML annotation
        tree = ET.parse(xml_path)
        t_root = tree.getroot()
        
        # get bounding box coordinates
        boxes = []
        for obj in t_root.findall('object'):
            bnd_box = obj.find('bndbox')
            xmin = float(bnd_box.find('xmin').text)
            xmax = float(bnd_box.find('xmax').text)
            ymin = float(bnd_box.find('ymin').text)
            ymax = float(bnd_box.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) 
        
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])

        # area of the bounding box, used during evaluation with the COCO metric for small, medium and large boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
      
        return img, target
    
    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Object Detection Training')
    parser.add_argument('--data_path', 
                        default='./BuildData/', help='the path to the dataset')
    parser.add_argument('--batch_size', 
                        default=2, type=int)
    parser.add_argument('--epochs', 
                        default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', 
                        default=4, type=int, help='number of data loading workers')
    parser.add_argument('--learning_rate', 
                        default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', 
                        default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', 
                        default=0.0005, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_step_size', 
                        default=3, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr_gamma', 
                        default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print_freq', 
                        default=10, type=int, help='print frequency')
    parser.add_argument('--output_dir', 
                        default='outputs', help='path where to save')
    parser.add_argument('--anchor_sizes', 
                        default=32, type=int, nargs='+', help='anchor sizes')
    parser.add_argument('--anchor_aspect_ratios', 
                        default= 1.0, type=float, nargs='+', help='anchor aspect ratios')
    parser.add_argument('--rpn_nms_thresh', 
                        default= 0.7, type=float,  help='NMS threshold used for postprocessing the RPN proposals')
    parser.add_argument('--box_nms_thresh', 
                        default= 0.5, type=float,  help='NMS threshold for the prediction head. Used during inference')
    parser.add_argument('--box_score_thresh', 
                        default= 0.05, type=float,  help='during inference only return proposals' 
                        'with a classification score greater than box_score_thresh')
    parser.add_argument('--box_detections_per_img', 
                        default= 100, type=int,  help='maximum number of detections per image, for all classes')
    args = parser.parse_args() 

data_path = args.data_path

# use our dataset and defined transformations
dataset = BuildDataset(data_path, get_transform(train=True))
dataset_test = BuildDataset(data_path, get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-100])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

batch_size = args.batch_size
workers = args.workers

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=workers,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=workers,
    collate_fn=utils.collate_fn)

# our dataset has two classes only - background and out of stock
num_classes = 2

rpn_nms_threshold = args.rpn_nms_thresh
box_nms_threshold = args.box_nms_thresh
box_score_threshold = args.box_score_thresh
num_box_detections = args.box_detections_per_img

# load pre-trained maskRCNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, rpn_nms_thresh=rpn_nms_threshold,
                                                           box_nms_thresh=box_nms_threshold, box_score_thresh=box_score_threshold,
                                                           box_detections_per_img=num_box_detections)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

anchor_sizes = tuple(args.anchor_sizes)
anchor_aspect_ratios = tuple(args.anchor_aspect_ratios)

# create an anchor_generator for the FPN which by default has 5 outputs
anchor_generator = AnchorGenerator(
    sizes=tuple([anchor_sizes for _ in range(5)]),
    aspect_ratios=tuple([anchor_aspect_ratios for _ in range(5)]))
model.rpn.anchor_generator = anchor_generator

# get number of input features for the RPN returned by FPN (256)
in_channels = model.backbone.out_channels

# replace the RPN head 
model.rpn.head = RPNHead(in_channels, anchor_generator.num_anchors_per_location()[0])

# turn off masks since dataset only has bounding boxes
model.roi_heads.mask_roi_pool = None

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

learning_rate = args.learning_rate
momentum = args.momentum
weight_decay = args.weight_decay

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate,
                            momentum=momentum, weight_decay=weight_decay)

lr_step_size = args.lr_step_size
lr_gamma = args.lr_gamma

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=lr_step_size,
                                               gamma=lr_gamma)

# number of training epochs
num_epochs = args.epochs
print_freq = args.print_freq

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset after every epoch
    evaluate(model, data_loader_test, device=device)

print("That's it!")
