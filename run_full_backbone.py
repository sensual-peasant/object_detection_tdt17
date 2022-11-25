# %% [markdown]
# ### Miniproject Henrik Ågotnes

# %%
import numpy as np
import time
import math
import sys
import code
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from skimage import io, transform

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
print('modules imported')


_path = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/annotations/xmls/Norway_002967.xml'
xml_paths = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/annotations/xmls'
image_path = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/images'






class RoadDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True,transform=None, country='Norway'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/' + country +'/'
        self.path_train_or_test = 'train' if train else 'test'
        self.country = country
        self.n_samples = 8160 if train else 2040
        if country in {'Czech': 2829, 'United_States': 4805, 'Japan':10506}:
            self.n_samples = {'Czech': 2829, 'United_States': 4805, 'Japan':10506}[country]
        
        self.image_dir = os.path.join(self.root_dir,self.path_train_or_test, 'images')
        self.annotation_dir = os.path.join(self.root_dir,self.path_train_or_test, 'annotations', 'xmls')
        

        #self.root_dir = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/'
        #self.image_dir = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/images/'
        #self.annotation_dir = '/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/annotations/xmls/'
        self.transform = transform

        self.img_shape = (3, 2044, 3650)
        self.num_classes = 4

        self.transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
        self.crack_name_to_label = {
            'D00': 1,
            'D10': 2,
            'D20': 3,
            'D40': 4,
        }
        self.num_classes = 4
        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.path_train_or_test == 'test':
            idx = idx + 8161

        img_name = os.path.join(self.image_dir, self.country + '_'
                                 + f'{idx:06d}' + '.jpg')
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.path_train_or_test == 'test':
            return image, {'image_id': idx}

        annotation_name = os.path.join(self.annotation_dir, self.country +'_' + f'{idx:06d}' + '.xml')
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

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        if boxes.shape[0] > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)
        else:
            area = torch.tensor([])

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx]), 'iscrowd':iscrowd, 'area':area}



        return image, target
    
    def get_pil(self, idx):
        img_name = os.path.join(self.image_dir,
                                'Norway_' + f'{idx:06d}' + '.jpg')
        return Image.open(img_name).convert("RGB") # transforms.functional.pil_to_tensor(




def collate_fn(batch):
    return tuple(zip(*batch))
road_dataset = RoadDataset(train=True)

train_size = int(len(road_dataset) * 0.75)
val_size = len(road_dataset) - train_size
train_size, val_size
train_data, val_data = torch.utils.data.random_split(road_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
test_data = RoadDataset(train=False)

train_data_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=1,
 collate_fn=collate_fn)
val_data_loader = DataLoader(val_data, batch_size=2, shuffle=True, num_workers=1,
 collate_fn=collate_fn)
test_data_loader = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=1,
 collate_fn=collate_fn)


 
#us_czech_japan_dataset = torch.utils.data.ConcatDataset([RoadDataset(train=True, country='United_States'), RoadDataset(train=True, country='Czech'),  RoadDataset(train=True, country='Japan')  ])

pretrain_data_loader = DataLoader(RoadDataset(train=True, country='United_States'), batch_size=2, shuffle=True, num_workers=1,
 collate_fn=collate_fn)


import cv2

def visualize():
    """
    draw the box. very, slow, avoid using
    """
    images, targets = next(iter(train_data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    for i in range(len(images)):
        if targets[i]['boxes'].shape[0]== 0:   continue
        boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
        sample = images[i].permute(1,2,0).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        for box in boxes:
            cv2.rectangle(sample,
                        (box[0], box[1]),
                        (box[2], box[3]),
                        (220, 0, 0), 3)
        ax.set_axis_off()
        plt.imshow(sample)
        plt.show()
        return targets[i]['labels']
#visualize()

# %% [markdown]
# ### Make model

# %%
# Problems with packages.
# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, pretrained= True)  # load pretrained


# %%
# Model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
def get_backbone_model():
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((4, 8, 32, 128, 256),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                    num_classes=5,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    return model
# load a model pre-trained pre-trained on COCO
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 5  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # move model to the right device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device) # TODO Change to device when running!!
    return model
#model.to('cpu') # TODO Change to device when running!!
model = get_backbone_model()#get_backbone_model()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# %% [markdown]
# ### Test forward

# %%
def test_forward(device):
    images,targets = next(iter(train_data_loader))
    images = [image.to(device) for image in images]
    model.eval()
    return model(images)

#out = test_forward('cuda')
#out



# %% [markdown]
# ### Train loop

# %%
def train(train_dataloader):
    model.train()
    running_loss = 0
    print(' Iterating...')
    for i, data in enumerate(train_dataloader):
        
        optimizer.zero_grad()
        images, targets = data[0], data[1]

        x, y = ([], [])
        for image, target in zip(images, targets):
            if len(target['labels']) > 0:
                x.append(image.to(device))
                y.append({k: v.to(device) for k, v in target.items()}) # [{k: v.to(device) for k, v in t.items()} for t in targets]
        if len(x) <= 0:
            continue
        #images = list(image.to(device) for image in images)
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if i % 25 == 0 or i<3:
            print(f"Iteration #{i}", end=' ')
        loss_dict = model(x, y)

        loss = sum(loss for loss in loss_dict.values())
        if i % 25 == 0 or i<3:
                    print(f"loss: {loss}")
         
        if not math.isfinite(loss):
            print(f"Loss is {loss}, stopping training")
            print('DEBUG! DEBUG DEBUG thats weird...')
            print(y)
            print(x)
            print('loss dict:\n', loss_dict)
            print('DEBUG END')
            optimizer.zero_grad()
            continue
            sys.exit(1)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        if i % 400 == 0 and i > 0:
            save_model_path = f'saved_models/test_e{epoch}_batch_{i}.pt'
            torch.save(model.state_dict(), save_model_path)
            print('Saved model to  ', save_model_path)
    train_loss = running_loss/len(train_dataloader.dataset)
    return train_loss


print('starting PRETRAINING!')
for epoch in range(10):  # # runnn train
    start = time.time()
    train_loss = train(pretrain_data_loader)
    print(f"Epoch #{epoch} loss: {train_loss}")
    end = time.time()
    print(f"Took {(end - start) / 60} minutes for epoch {epoch}")

    save_model_path = f'saved_models_2/test_e{epoch}_batch_end_of_batch.pt'
    torch.save(model.state_dict(), save_model_path)
    print('Saved model to  ', save_model_path)


print('starting actual training!')
for epoch in range(12):  # # runnn train
    start = time.time()
    train_loss = train(train_data_loader)
    print(f"Epoch #{epoch} loss: {train_loss}")
    end = time.time()
    print(f"Took {(end - start) / 60} minutes for epoch {epoch}")

    save_model_path = f'saved_models_2/test_e{epoch}_batch_{0}_job_after_batch.pt'
    torch.save(model.state_dict(), save_model_path)
    print('Saved model to  ', save_model_path)


sys.exit()
### load model and test it
device = 'cuda'
torch.cuda.empty_cache()
model = get_backbone_model().to(device)

PATH = 'saved_models/test_e9_batch_1800.pt' # 'saved_models/test.pt'
model.load_state_dict(torch.load(PATH))


def test_forward(device):
    with torch.no_grad():
        images,targets = next(iter(train_data_loader))
        images = [image.to(device) for image in images]
        model.eval()
        return model(images)
print('Testing loaded model...')
out = test_forward(device)
print(out)
print(out[0]['boxes'].shape)


### eval attempt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
iou_tresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
metric = MeanAveragePrecision(iou_thresholds=iou_tresholds)


def evaluate():
    device = 'cuda'
    model.to(device)
    model.eval()
    evals = []


    for images, targets in val_data_loader: # val_data_loader
        x, y = ([], [])
        for image, target in zip(images, targets):
            if len(target['labels']) > 0:
                x.append(image.to(device))
                y.append({k: v.to(device) for k, v in target.items()}) # [{k: v.to(device) for k, v in t.items()} for t in targets]
        if len(x) <= 0:
            continue
        
        with torch.no_grad():
            model_time = time.time()
            preds = model(x)
            
            #print(preds)
            model_time = time.time() - model_time
            evals.append(model_time)
                
            metric.update(preds, y)
            
            #
            evals.append(model_time)
            if len(evals) % 100 ==0: # todo remove to run whole thing
                print(f'progress: {round(100*len(evals)/ len(train_data_loader), 2)}% values out of {len(evals), len(train_data_loader)}')
                #print(' Following map: ', metric.compute())
            

    print(f'Average running time: {sum(evals)/len(evals)}')    
    #map_dict = metric.compute()
    return metric
evaluate() # runnn eval
with torch.no_grad():
    map = metric.compute()


print(f'Most important: metric: {PATH, map}')
with open('/cluster/home/henrikya/object_detection/knegg.pdf','a') as fd:

    # a_str = 'map: ' + str(map['map']) + 'map_50: ' + str(map['map_50']) + ',   map_75: ' + str(map['map_75']) + str(map['map_medium']) + str(map['map_large']) # map_medium, map_large
    fd.write(PATH+str(map))

# try saving map to file:

### submission attempt
class Submission:
    """
    From chat:
    Norway_000000.jpg, 0 1 2 3 4 5
    
    ImageId: the id of the test image, for example, India_00001
    PredictionString: the prediction string should be space-delimited of 5 integers. For example, 2 240 170 260 240 means
    it's label 2, with a bounding box of coordinates (x_min, y_min, x_max, y_max). We accept up to 5 predictions. 
    For example, if you submit: 
    3 42 24 170 186 1 292 28 430 198 4 168 24 292 190 5 299 238 443 374 2 160 195 294 357 6 1 224 135 356 
    which contains 6 bounding boxes, we will only take the first 5 into consideration.
    """
    
    def __init__(self) -> None:
        self.objectness_treshold = 0.8

        self.n_preds = len(test_data)

        self.submission = pd.DataFrame({'ImageId':['INVALID' for i in range(self.n_preds)], 'PredictionString':['INVALID' for i in range(self.n_preds)]})

        self.submission.index = [i + 8161 for i in range(self.n_preds)]

    def add_batch_preds(self, preds, targets):
        for pred, target in zip(preds, targets):
            idx = target['image_id']
            self.submission.loc[idx, 'ImageId'] = f'Norway_{idx:06d}.jpg'
            # use threshold
            # select only 5 highest above threshold
            mask = pred['scores'].data.numpy() > self.objectness_treshold
            boxes = pred['boxes'].data.numpy()[mask].astype(int)
            labels = pred['labels'].data.numpy()[mask]

            # sanity check
            if len(labels) != boxes.shape[0]:
                print('thats weird!')
                print(boxes, labels, targets, preds)

            a_str = ''
            for i in range(len(labels)):
                a_str += str(labels[i])+ ' ' +' '.join(str(x) for x in boxes[i, :]) + ' ' 
            
            
            self.submission.loc[idx, 'PredictionString'] = a_str
    
    def save_preds(self):
        self.submission.to_csv('/cluster/home/henrikya/object_detection/predictions/submission.csv', index=False, header=False)


def fill_submission():
    device = 'cpu'
    model.to(device)
    model.eval()
    evals = []

    submission = Submission()

    for x, targets in test_data_loader: # val_data_loader
        x = [image.to(device) for image in x]
        
        with torch.no_grad():
            model_time = time.time()
            preds = model(x)
            print('len', targets, len(preds))
            print(preds)
            model_time = time.time() - model_time

            submission.add_batch_preds(preds, targets)
            

        evals.append(model_time)
        # # TODO REMOVE
    print('average model time:', sum(evals)/len(evals))
    submission.save_preds()
    return submission 
#submission = fill_submission() # runnn test

