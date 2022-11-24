# %% [markdown]
# ### Miniproject Henrik Ågotnes

# %%
import numpy as np
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
print('modules imported')


_path = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/annotations/xmls/Norway_002967.xml'
xml_paths = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/annotations/xmls'
image_path = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/images'






class RoadDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/'
        self.path_train_or_test = 'train' if train else 'test'
        self.n_samples = 8160 if train else 2040
        self.image_dir = os.path.join(self.root_dir,self.path_train_or_test, 'images')
        self.annotation_dir = os.path.join(self.root_dir,self.path_train_or_test, 'annotations', 'xmls')
        

        #self.root_dir = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/'
        #self.image_dir = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/images/'
        #self.annotation_dir = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/annotations/xmls/'
        self.transform = transform

        self.img_shape = (3, 2044, 3650)
        self.num_classes = 5

        self.transform = transforms.Compose([transforms.ToTensor()]) #[transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
        self.crack_name_to_label = {
            'D00': 1,
            'D10': 2,
            'D20': 3,
            'D40': 4,
        }
        self.num_classes = 5
        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.path_train_or_test == 'test':
            idx = idx + 8161

        img_name = os.path.join(self.image_dir,
                                'Norway_' + f'{idx:06d}' + '.jpg')
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.path_train_or_test == 'test':
            return image, {'image_id': idx}

        annotation_name = os.path.join(self.annotation_dir, 'Norway_' + f'{idx:06d}' + '.xml')
        root = ET.parse(annotation_name).getroot()
        labels, boxes = ([], [])
        for _object in root.findall("object"):
            name = _object.find('name').text
            if name in self.crack_name_to_label:
                labels.append(self.crack_name_to_label[name])
            #labels.append(self.crack_name_to_label[_object.find('name').text])
            boxes.append([float(i.text) for i in _object.find('bndbox')])
            if _object.find('name').text not in self.crack_name_to_label:
                print('[WARNING]Found where object name:', _object.find('name').text)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}



        return image, target
print('def Road')


# %%
def collate_fn(batch):
    return tuple(zip(*batch))
road_dataset = RoadDataset(train=True)

train_size = int(len(road_dataset) * 0.75)
val_size = len(road_dataset) - train_size
train_size, val_size
train_data, val_data = torch.utils.data.random_split(road_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
test_data = RoadDataset(train=False)

train_data_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4,
 collate_fn=collate_fn)
val_data_loader = DataLoader(val_data, batch_size=2, shuffle=True, num_workers=4,
 collate_fn=collate_fn)
test_data_loader = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=4,
 collate_fn=collate_fn)

print('loading data')
# %% [markdown]
# ### Make model

# %%
# Model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 5  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# move model to the right device


# load model
PATH = '/cluster/home/henrikya/object_detection/saved_models/backup/torch_e0_batch_237.pt'
#model.load_state_dict(torch.load(PATH))
#
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device) # TODO Change to device when running!!

# %% [markdown]
# ### Test forward
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# %% [markdown]
# ### Train loop



# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# %%
""" training loop global.
model.train()
model.to(device)
start =time.time()
for epoch in range(10):
    print(f'[Epoch: {epoch}]:')
    for batch_i, (images, targets) in enumerate(train_data_loader):
        
        x, y = ([], [])
        for image, target in zip(images, targets):
            if len(target['labels']) > 0:
                x.append(image.to(device))
                y.append({k: v.to(device) for k, v in target.items()}) # [{k: v.to(device) for k, v in t.items()} for t in targets]
        if len(x) <= 0:
            continue

         
        loss_dict = model(x, y)
        losses = sum(loss for loss in loss_dict.values())
        if batch_i % 50==0:
            print(f' [{batch_i}]  Losses are: {losses=}, Other: {batch_i=}, {type(images)}, {type(targets)}')   
        # TODO report loss
        
        losses.backward()
        optimizer.step()
        lr_scheduler.step()
        if time.time() - start > 1800.0:
            break
    save_model_path = os.path.join( '/cluster/home/henrikya/object_detection/saved_models/', f'torch_e{epoch}_batch_{batch_i}.pt')
    print(f'Saving to: {save_model_path}')
    torch.save(model.state_dict(), save_model_path)
    if time.time() - start > 1800.0:
            break

"""
def train(train_data_loader):
    model.train()
    model.to(device)
    start =time.time()
    for epoch in range(30):
        print(f'[Epoch: {epoch}]:')
        for batch_i, (images, targets) in enumerate(train_data_loader):
            
            x, y = ([], [])
            for image, target in zip(images, targets):
                if len(target['labels']) > 0:
                    x.append(image.to(device))
                    y.append({k: v.to(device) for k, v in target.items()}) # [{k: v.to(device) for k, v in t.items()} for t in targets]
            if len(x) <= 0:
                continue
            
            #print(f'    [{batch_i}] time: {round(time.time() - start)}')
            
            loss_dict = model(x, y)
            losses = sum(loss for loss in loss_dict.values())
            if batch_i % 100==0:
                print(f' [{batch_i}]  Losses are: {losses=}, Try to save: {batch_i=}, {type(images)}, {type(targets)}')
                save_model_path = os.path.join( '/cluster/home/henrikya/object_detection/saved_models/', f'torch_e{epoch}_batch_{batch_i}.pt')
                print(f'Saving to: {save_model_path}')
                torch.save(model.state_dict(), save_model_path)
                print(f' hmmm, success?')
            # TODO report loss
            
            losses.backward()
            optimizer.step()
            lr_scheduler.step()
        save_model_path = os.path.join( '/cluster/home/henrikya/object_detection/saved_models/', f'torch_e{epoch}_batch_{batch_i}.pt')
        print(f'Saving to: {save_model_path}')
        torch.save(model.state_dict(), save_model_path)
        
# deal with metrics
print('train...')
train(train_data_loader)
"""
from torchmetrics.detection.mean_ap import MeanAveragePrecision
iou_tresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
metric = MeanAveragePrecision(iou_thresholds=iou_tresholds)

model.eval()
model.to('cpu')
evals = []
for images, targets in val_data_loader:
    model_time = time.time()
    preds = model(images)
    model_time = time.time() - model_time
    metric.update(list(preds), list(targets))


map_dict = metric.compute()
print('results:', map_dict)"""
def eval():
    import os, psutil
    process = psutil.Process(os.getpid())
    #
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    iou_tresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    metric = MeanAveragePrecision(iou_thresholds=iou_tresholds)

    model.eval()
    model.to('cpu')
    evals = []

    batch_i = 0
    for images, targets in val_data_loader:
        model_time = time.time()
        preds = model(images)
        model_time = time.time() - model_time
        #metric.update(list(preds), list(targets))

        if batch_i % 2 == 0:
            print(f'{batch_i}, {metric.compute()}')
        print('mem usage 9 normal): ', round(process.memory_info().rss/1000000, 2))
        batch_i +=1


    map_dict = metric.compute()
    print('results:', map_dict)

#eval()