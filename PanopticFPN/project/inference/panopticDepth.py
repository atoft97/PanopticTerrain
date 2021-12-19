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
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from os import listdir

from torchvision.transforms import Compose

from detectron2.modeling import build_model

import json
from detectron2.data.datasets import register_coco_panoptic_separated

model_name = "fpn101CocoShort"

device = torch.device("cuda")

register_coco_panoptic_separated(name="ffi_train", sem_seg_root="dataset_train/panoptic_stuff_train", metadata={}, image_root="dataset_train/images", panoptic_root="dataset_train/panoptic_train" , panoptic_json="dataset_train/annotations/panoptic_train.json" ,instances_json="dataset_train/annotations/instances_train.json")
register_coco_panoptic_separated(name="ffi_test", sem_seg_root="dataset_test/panoptic_stuff_test", metadata={}, image_root="dataset_test/images", panoptic_root="dataset_test/panoptic_test" , panoptic_json="dataset_test/annotations/panoptic_test.json" ,instances_json="dataset_test/annotations/instances_test.json")

# kanskje legg til instance segmentaiotion fil

MetadataCatalog.get("ffi_train_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'CameraEdge','Bush', 'Grass',  'Gravel', ])
MetadataCatalog.get("ffi_train_separated").set(stuff_colors=[[255,255,255], [15,171,255], [191,140,0], [46,153,0], [70,70,70], [232,227,81], [64,255,38], [180,180,180], ])
#MetadataCatalog.get("ffi_test_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 5: 4, 10: 5, 11: 6, 12: 7})
MetadataCatalog.get("ffi_train_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 2: 4, 5: 5, 10: 6, 11: 7})


MetadataCatalog.get("ffi_train_separated").set(thing_classes=['Vehicle','Person', 'Puddle', 'Large_stone', 'Tree', 'Building'])
MetadataCatalog.get("ffi_train_separated").set(thing_colors=[[150,0,191],[255,38,38], [255,179,0], [200,200,200], [118,255,99], [255,20,20]])


#cfg.merge_from_file("../configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
MetadataCatalog.get("ffi_test_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'CameraEdge','Bush', 'Grass',  'Gravel', ])
MetadataCatalog.get("ffi_test_separated").set(stuff_colors=[[255,255,255], [191,140,0], [15,171,255], [46,153,0], [70,70,70], [232,227,81], [64,255,38], [180,180,180], ])
#MetadataCatalog.get("ffi_test_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 5: 4, 10: 5, 11: 6, 12: 7})
MetadataCatalog.get("ffi_test_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 2: 4, 5: 5, 10: 6, 11: 7})

MetadataCatalog.get("ffi_test_separated").set(thing_classes=['Vehicle','Person', 'Puddle', 'Large_stone', 'Tree', 'Building'])
MetadataCatalog.get("ffi_test_separated").set(thing_colors=[[150,0,191],[255,38,38], [255,179,0], [200,200,200], [118,255,99], [255,20,20]])
MetadataCatalog.get("ffi_test_separated").set(thing_dataset_id_to_contiguous_id={3: 0, 4: 1, 6: 2, 9: 3, 12: 4, 13: 5})

cfg = get_cfg()

cfg.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
cfg.MODEL.WEIGHTS = "../training/models/" + model_name + "/"+ model_name + ".pth"

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8
cfg.DATASETS.TEST = ("ffi_test_separated", )
cfg.freeze()


metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)
predictor = DefaultPredictor(cfg)


#typeOfFrame = "Video"
typeOfFrame = "Image"
#typeOfFrame = "VideoOptak"

index = 0
startPath = "dataset_test/images"

files = listdir(startPath)
files.sort()
print(files)

saving = True

if (typeOfFrame == "Video"):
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
elif (typeOfFrame == "VideoOptak"):
    #cam = cv2.VideoCapture("/home/potetsos/skule/2021Host/prosjektTesting/combined/detectron2/demo/videoer/Kia-eNiro-AV_LowRes.mp4")
    cam = cv2.VideoCapture("/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/filmer/rockQuarryIntoWoodsDrive.mp4")
    #cam = cv2.VideoCapture("/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/stille.mp4")
    num_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    #codec, file_ext = ("x264", ".mkv")
    codec, file_ext = ("mp4v", ".mp4")
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fname = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/rockQuarryIntoWoodsDrive_Analyzed" + file_ext
    frames_per_second = cam.get(cv2.CAP_PROP_FPS)
    print("FPS", frames_per_second)
    output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    #frameSize=(3840, 1080),
                    frameSize=(4096, 1536),
                    #frameSize=(2048, 540),
                    isColor=True,
            )
    print("\n")
    print(output_file)
    print("\n")

def getFrame(index):
    if (typeOfFrame == "Video" or typeOfFrame == "VideoOptak"):
        success, frame = cam.read()
    if (typeOfFrame == "Image"):
        print(str(index) + "/" +str(len(files)))
        frame = read_image(startPath + "/" +files[index], format="BGR")
        index+=1
    return(frame)

def visualise_predicted_frame(frame, predictions):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if "panoptic_seg" in predictions:
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        vis_frame = video_visualizer.draw_panoptic_seg_predictions(
            frame, panoptic_seg.to('cpu'), segments_info
        )
    elif ("instances" in predictions):
        predictions = predictions["instances"].to(device)
        vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)

    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return(vis_frame)

data = DatasetCatalog.get('ffi_test_separated')

model_name = "fasit"
files.sort()

counter = 0
totalTime = 0
for fileName in files:
    counter += 1
    print(str(counter) + "/" +str(len(files)))
    print(data[counter-1]['file_name'])
    newFIleName = data[counter-1]['file_name']
    frame = read_image(newFIleName, format="BGR")
    newFIleName = newFIleName[len("dataset_test/images/"):]
    startTime = time.time()
    predictedPanoptic = predictor(frame)
    diffTime = time.time() - startTime
    totalTime += diffTime
    print("avgTime", totalTime/counter)

    #print(data[counter])
    visulizer = Visualizer(frame, metadata)
    out = visulizer.draw_dataset_dict(data[counter-1]).get_image()
    combinedFrame = np.hstack((out, frame))

    #vis_panoptic = visualise_predicted_frame(frame, predictedPanoptic)
    #combinedFrame = np.hstack((vis_panoptic, frame))

    if (saving == True):
        cv2.imwrite("output/" + model_name + "/" + newFIleName, combinedFrame)

