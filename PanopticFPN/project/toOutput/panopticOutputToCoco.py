import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer, _PanopticPrediction
import torch
from os import listdir

from torchvision.transforms import Compose

from detectron2.modeling import build_model

import json

from detectron2.checkpoint import DetectionCheckpointer



import detectron2.data.transforms as T
import matplotlib.pyplot as plt

import pycocotools.mask as mask_util

device = torch.device("cpu")


cfg = get_cfg()
cfg.merge_from_file("../training/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
#cfg.merge_from_list(['MODEL.DEVICE', 'cpu', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])
cfg.MODEL.WEIGHTS = "../training/outputMultiGPU/model_final.pth"
cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

MetadataCatalog.get("ffi_test_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'CameraEdge','Bush', 'Grass',  'Gravel', ])
MetadataCatalog.get("ffi_test_separated").set(stuff_colors=[[255,255,255], [15,171,255], [191,140,0], [46,153,0], [70,70,70], [232,227,81], [64,255,38], [180,180,180], ])
MetadataCatalog.get("ffi_test_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 2: 4, 5: 5, 10: 6, 11: 7})

MetadataCatalog.get("ffi_test_separated").set(thing_classes=['Vehicle','Person', 'Puddle', 'Large_stone', 'Tree', 'Building'])
MetadataCatalog.get("ffi_test_separated").set(thing_colors=[[150,0,191],[255,38,38], [255,179,0], [200,200,200], [118,255,99], [255,20,20]])
MetadataCatalog.get("ffi_test_separated").set(thing_dataset_id_to_contiguous_id={3: 0, 4: 1, 6: 2, 9: 3, 12: 4, 13: 5})
cfg.DATASETS.TEST = ("ffi_test_separated", )
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8
cfg.INPUT.MASK_FORMAT = "bitmask"


cfg.freeze()

metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
WINDOW_NAME = "WebCamTest"
video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)
predictor = DefaultPredictor(cfg)



model = build_model(cfg)
model.eval()

checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)


aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

input_format = cfg.INPUT.FORMAT



def predictImage(frame):

    with torch.no_grad():
    	height, width = frame.shape[:2]
    	image = aug.get_transform(frame).apply_image(frame)
    	image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    	inputs = {"image": image, "height": height, "width": width}
    	
    #	print(inputs)
    	predictions = model([inputs])
    	predictions = predictions[0]

    predictedPanoptic = predictions

    return(predictedPanoptic)


def detectronToCoco(predictedPanoptic, iamgeID, startID):
    panoptic_seg, segments_info = predictedPanoptic["panoptic_seg"]
    tensors = []
    for segment_number in range(1, (len(segments_info) +1)): #range på alle klassa
        #print(pan_class)
        class_tensor = torch.where(panoptic_seg == segment_number, 1, 0)
        tensors.append(class_tensor)

    stacked = torch.stack(tensors)
    stackedNumpy = stacked.cpu().detach().numpy()

    number=10

    annotations = []



    for segment_number in range((len(segments_info))):
        segmentDict = {'id': segment_number+startID}

        thing_id = dict((v,k) for k,v in metadata.thing_dataset_id_to_contiguous_id.items())
        stuff_id = dict((v,k) for k,v in metadata.stuff_dataset_id_to_contiguous_id.items())

        cat_id = segments_info[segment_number]['category_id']


        if (segments_info[segment_number]['isthing']):
            segmentDict['category_id'] = thing_id[cat_id]
        else:
            segmentDict['category_id'] = stuff_id[cat_id]

        
        segmentDict['image_id'] = iamgeID
        segmentDict['area'] = 0
        segmentDict['bbox'] = [0,0,0,0]
        segmentDict['iscrowd'] = 0

        contours, hierarchy = cv2.findContours((stackedNumpy[segment_number]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation_list = []


        for countor in contours:
            epsilon = 0.0003 * cv2.arcLength(countor, True)
            approximations = cv2.approxPolyDP(countor, epsilon, True)

            x_y_list = []
            for coordinates in approximations:
                x_y_list.append(int(coordinates[0][0]))
                x_y_list.append(int(coordinates[0][1]))
            if (len(x_y_list)>4):
                segmentation_list.append(x_y_list)

        segmentDict['segmentation'] = segmentation_list
        annotations.append(segmentDict)
    return(annotations)


def metadatToCoco(metadata):
    categories = []

    thing_id = dict((v,k) for k,v in metadata.thing_dataset_id_to_contiguous_id.items())
    for i in range(len(metadata.thing_classes)):
        category = {'id': thing_id[i], 'name': metadata.thing_classes[i], "supercategory": ""}
        categories.append(category)

    stuff_id = dict((v,k) for k,v in metadata.stuff_dataset_id_to_contiguous_id.items())
    for i in range(1, len(metadata.stuff_classes)):
        category = {'id': stuff_id[i], 'name': metadata.stuff_classes[i], "supercategory": ""}
        categories.append(category)



    return(categories)


def visualise_predicted_frame(frame, predictions):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if "panoptic_seg" in predictions:
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        vis_frame = video_visualizer.draw_panoptic_seg_predictions(
            frame, panoptic_seg.to(device), segments_info
        )
    elif ("instances" in predictions):
        predictions = predictions["instances"].to(device)
        vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)

    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

    return(vis_frame)


coco = {"licenses": [
    {
      "name": "",
      "id": 0,
      "url": ""
    }],  
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": ""
  },
  }



categories = metadatToCoco(metadata)
coco['categories'] = categories


#frame = read_image("bilder/ffiKjoretoy.jpg", format="BGR")
startPath = "testDiv"
files = listdir(startPath)



'''
    "images": [
    {
      "id": 1,
      #"width": 1214,
      #"height": 910,
      #"file_name": "3Kara.jpg",
      "width": 800,
      "height": 533,
      "file_name": "ffiKjoretoy.jpg",
      "license": 0,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": 0
    }
    ],
'''
images = []
counter = 1
segmentID = 1
annotations = []
for imageName in files:
    print(counter, "/", len(files))
    frame = read_image(startPath + "/" +imageName, format="BGR")

    predictedPanoptic = predictImage(frame)

    vis_panoptic = visualise_predicted_frame(frame, predictedPanoptic)
    combinedFrame = np.hstack((vis_panoptic, frame))
    cv2.imwrite("output/testDiv/" + imageName, combinedFrame)

    height, width = frame.shape[:2]
    imageCoco = {"id": counter, "width": width, "height": height, "file_name": imageName, "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}
    
    images.append(imageCoco)
    #skriv ut coco til samme mappa

    annotationsImage = detectronToCoco(predictedPanoptic, counter, segmentID)
    annotations.extend(annotationsImage)
    segmentID += len(annotationsImage)

    counter+= 1

coco['images'] = images

#annotations = detectronToCoco(predictedPanoptic, 1)
coco['annotations'] = annotations


with open('anotationsTestDiv.json', 'w') as fp:
    json.dump(coco, fp)








