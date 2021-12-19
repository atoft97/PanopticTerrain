from torch._C import device
import detectron2
import numpy as np
import os
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets import register_coco_panoptic
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor

from detectron2.evaluation.panoptic_evaluation import COCOPanopticEvaluator
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling import build_model
from detectron2.evaluation import inference_on_dataset

from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import read_image
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data import build_detection_test_loader

import torch

from detectron2.projects.deeplab import add_deeplab_config
from mask_former import add_mask_former_config

#print(torch.cuda.memory_summary(device=None, abbreviated=False))
cfg = get_cfg()
add_deeplab_config(cfg)
add_mask_former_config(cfg)
#cfg.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
cfg.merge_from_file("configs/coco-panoptic/swin/maskformer_panoptic_swin_large_IN21k_384_bs64_554k.yaml")
cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'models/model_final_7505c4.pkl'])

#thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

#register_coco_panoptic("my_dataset_train", {}, "val2017","annotations/panoptic_val2017", "annotations/panoptic_val2017.json", instances_json="annotations/annotations/instances_val2017")
register_coco_panoptic_separated(name="ffi_train", sem_seg_root="dataset_train/panoptic_stuff_train", metadata={}, image_root="dataset_train/images", panoptic_root="dataset_train/panoptic_train" , panoptic_json="dataset_train/annotations/panoptic_train.json" ,instances_json="dataset_train/annotations/instances_train.json")
register_coco_panoptic_separated(name="ffi_test", sem_seg_root="dataset_test/panoptic_stuff_test", metadata={}, image_root="dataset_test/images", panoptic_root="dataset_test/panoptic_test" , panoptic_json="dataset_test/annotations/panoptic_test.json" ,instances_json="dataset_test/annotations/instances_test.json")

# kanskje legg til instance segmentaiotion fil

print(MetadataCatalog.get("ffi_train_separated"))
#print(MetadataCatalog.get("coco_2017_train_panoptic_separated"))
'''
[
{"supercategory": "Person",     "color": [255,38,38],   "isthing": 1, "id": 4,  "name": "Person"},
{"supercategory": "Sky",        "color": [15,171,255],  "isthing": 0, "id": 8,  "name": "Sky"},
{"supercategory": "Dirtroad",   "color": [191,140,0],   "isthing": 0, "id": 7,  "name": "Dirtroad"},
{"supercategory": "Vehicle",    "color": [150,0,191],   "isthing": 1, "id": 3,  "name": "Vehicle"},
{"supercategory": "Forrest",    "color": [46,153,0],    "isthing": 0, "id": 1,  "name": "Forrest"},
{"supercategory": "CameraEdge", "color": [70,70,70],    "isthing": 1, "id": 2, "name": "CameraEdge"},
{"supercategory": "Bush",       "color": [232,227,81],  "isthing": 0, "id": 5,  "name": "Bush"},
{"supercategory": "Puddle",     "color": [255,179,0],   "isthing": 1, "id": 6,  "name": "Puddle"},
{"supercategory": "Large_stone","color": [200,200,200], "isthing": 1, "id": 9, "name": "Large_stone"},
{"supercategory": "Grass",      "color": [64,255,38],   "isthing": 0, "id": 10,  "name": "Grass"},
{"supercategory": "Tall_grass", "color": [255,251,38],  "isthing": 0, "id": 11,  "name": "Tall_grass"},
{"supercategory": "Gravel",     "color": [180,180,180], "isthing": 0, "id": 12,  "name": "Gravel"},
{"supercategory": "Tree",       "color": [118,255,99],  "isthing": 1, "id": 13,  "name": "Tree"},
{"supercategory": "Building",   "color": [255,20,20],   "isthing": 1, "id": 14,  "name": "Building"}
]
'''
MetadataCatalog.get("ffi_train_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'CameraEdge','Bush', 'Grass',  'Gravel', ])
MetadataCatalog.get("ffi_train_separated").set(stuff_colors=[[255,255,255], [15,171,255], [191,140,0], [46,153,0], [70,70,70], [232,227,81], [64,255,38], [180,180,180], ])
#MetadataCatalog.get("ffi_test_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 5: 4, 10: 5, 11: 6, 12: 7})
MetadataCatalog.get("ffi_train_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 2: 4, 5: 5, 10: 6, 11: 7})


MetadataCatalog.get("ffi_train_separated").set(thing_classes=['Vehicle','Person', 'Puddle', 'Large_stone', 'Tree', 'Building'])
MetadataCatalog.get("ffi_train_separated").set(thing_colors=[[150,0,191],[255,38,38], [255,179,0], [200,200,200], [118,255,99], [255,20,20]])
#MetadataCatalog.get("ffi_train_separated").set(thing_dataset_id_to_contiguous_id={2:0, 3:1, 4:2, 6:3, 9:4})
#MetadataCatalog.get("mitt_dataset_separated").set(thing_colors=[[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157], [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240], [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164], [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243], [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255], [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95], [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210], [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0], [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170], [0, 182, 199], [0, 165, 120], [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185], [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45], [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122], [191, 162, 208]])
#MetadataCatalog.get("mitt_dataset_separated").set(thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79})
#print(MetadataCatalog.get("mitt_dataset_separated"))

#MetadataCatalog.get("mitt_dataset_val_separated").set(stuff_classes=['things', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind', 'window', 'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor', 'pavement', 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'wall', 'rug'])
#MetadataCatalog.get("mitt_dataset_val_separated").set(stuff_colors=[[82, 18, 128], [255, 255, 128], [147, 211, 203], [150, 100, 100], [168, 171, 172], [146, 112, 198], [210, 170, 100], [92, 136, 89], [218, 88, 184], [241, 129, 0], [217, 17, 255], [124, 74, 181], [70, 70, 70], [255, 228, 255], [154, 208, 0], [193, 0, 92], [76, 91, 113], [255, 180, 195], [106, 154, 176], [230, 150, 140], [60, 143, 255], [128, 64, 128], [92, 82, 55], [254, 212, 124], [73, 77, 174], [255, 160, 98], [255, 255, 255], [104, 84, 109], [169, 164, 131], [225, 199, 255], [137, 54, 74], [135, 158, 223], [7, 246, 231], [107, 255, 200], [58, 41, 149], [183, 121, 142], [255, 73, 97], [107, 142, 35], [190, 153, 153], [146, 139, 141], [70, 130, 180], [134, 199, 156], [209, 226, 140], [96, 36, 108], [96, 96, 96], [64, 170, 64], [152, 251, 152], [208, 229, 228], [206, 186, 171], [152, 161, 64], [116, 112, 0], [0, 114, 143], [102, 102, 156], [250, 141, 255]], stuff_dataset_id_to_contiguous_id={92: 1, 93: 2, 95: 3, 100: 4, 107: 5, 109: 6, 112: 7, 118: 8, 119: 9, 122: 10, 125: 11, 128: 12, 130: 13, 133: 14, 138: 15, 141: 16, 144: 17, 145: 18, 147: 19, 148: 20, 149: 21, 151: 22, 154: 23, 155: 24, 156: 25, 159: 26, 161: 27, 166: 28, 168: 29, 171: 30, 175: 31, 176: 32, 177: 33, 178: 34, 180: 35, 181: 36, 184: 37, 185: 38, 186: 39, 187: 40, 188: 41, 189: 42, 190: 43, 191: 44, 192: 45, 193: 46, 194: 47, 195: 48, 196: 49, 197: 50, 198: 51, 199: 52, 200: 53, 0: 0})
##MetadataCatalog.get("mitt_dataset_val_separated").set(stuff_dataset_id_to_contiguous_id={92: 1, 93: 2, 95: 3, 100: 4, 107: 5, 109: 6, 112: 7, 118: 8, 119: 9, 122: 10, 125: 11, 128: 12, 130: 13, 133: 14, 138: 15, 141: 16, 144: 17, 145: 18, 147: 19, 148: 20, 149: 21, 151: 22, 154: 23, 155: 24, 156: 25, 159: 26, 161: 27, 166: 28, 168: 29, 171: 30, 175: 31, 176: 32, 177: 33, 178: 34, 180: 35, 181: 36, 184: 37, 185: 38, 186: 39, 187: 40, 188: 41, 189: 42, 190: 43, 191: 44, 192: 45, 193: 46, 194: 47, 195: 48, 196: 49, 197: 50, 198: 51, 199: 52, 200: 53, 0: 0})
#MetadataCatalog.get("mitt_dataset_val_separated").set(thing_classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
#MetadataCatalog.get("mitt_dataset_val_separated").set(thing_colors=[[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157], [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240], [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164], [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243], [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255], [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95], [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210], [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0], [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170], [0, 182, 199], [0, 165, 120], [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185], [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45], [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122], [191, 162, 208]])
#MetadataCatalog.get("mitt_dataset_val_separated").set(thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79})
print(MetadataCatalog.get("ffi_train_separated"))

cfg.DATASETS.TRAIN = ("ffi_train_separated")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = []

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8

cfg.INPUT.MASK_FORMAT = "bitmask"
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#DatasetCatalog.get('coco_2017_train_panoptic')
print(cfg.MODEL.ROI_HEADS.NUM_CLASSES)

print(cfg)

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
#trainer.train()
#metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
#print(m_train2017etadata)



#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
#predictor = DefaultPredictor(cfg)


#res = evaluator.evaluate()
#print(res)


MetadataCatalog.get("ffi_test_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'CameraEdge','Bush', 'Grass',  'Gravel', ])
MetadataCatalog.get("ffi_test_separated").set(stuff_colors=[[255,255,255], [15,171,255], [191,140,0], [46,153,0], [70,70,70], [232,227,81], [64,255,38], [180,180,180], ])
#MetadataCatalog.get("ffi_test_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 5: 4, 10: 5, 11: 6, 12: 7})
MetadataCatalog.get("ffi_test_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 2: 4, 5: 5, 10: 6, 11: 7})

MetadataCatalog.get("ffi_test_separated").set(thing_classes=['Vehicle','Person', 'Puddle', 'Large_stone', 'Tree', 'Building'])
MetadataCatalog.get("ffi_test_separated").set(thing_colors=[[150,0,191],[255,38,38], [255,179,0], [200,200,200], [118,255,99], [255,20,20]])
MetadataCatalog.get("ffi_test_separated").set(thing_dataset_id_to_contiguous_id={3: 0, 4: 1, 6: 2, 9: 3, 12: 4, 13: 5})



cfg2 = get_cfg()

cfg2.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
#cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])

cfg2.merge_from_list(['MODEL.DEVICE', 'cuda'])
cfg2.MODEL.WEIGHTS = "output/model_final.pth"
#cfg2.MODEL.WEIGHTS = "detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl"
cfg2.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg2.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
cfg2.DATASETS.TEST = ("ffi_test_separated", )

cfg2.MODEL.ROI_HEADS.NUM_CLASSES = 6 
cfg2.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8
cfg2.INPUT.MASK_FORMAT = "bitmask"

cfg2.freeze()

metadata = MetadataCatalog.get(cfg2.DATASETS.TEST[0])
print(metadata)


video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)
print(cfg2)
#model = build_model(cfg2)
#print(model)
predictor = DefaultPredictor(cfg2)

#frame = read_image("datasettVariert/coco/images/3Kara.jpg", format="BGR")
frame = read_image("dataset_train/images/stille_frame00001.jpg", format="BGR")
outputs = predictor(frame)


#print(frame)


#print(outputs)


frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
panoptic_seg, segments_info = outputs["panoptic_seg"]

print(segments_info)
print(panoptic_seg)


vis_frame = video_visualizer.draw_panoptic_seg_predictions(
            frame, panoptic_seg.to('cpu'), segments_info
        )
vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

cv2.imwrite("stille.png", vis_frame)

evaluator_train = COCOPanopticEvaluator(dataset_name="ffi_train_separated")
val_loader = build_detection_test_loader(cfg2, "ffi_train_separated")
print(inference_on_dataset(predictor.model, val_loader, evaluator_train))




evaluator_test = COCOPanopticEvaluator(dataset_name="ffi_test_separated")
val_loader = build_detection_test_loader(cfg2, "ffi_test_separated")
print(inference_on_dataset(predictor.model, val_loader, evaluator_test))


#res = trainer.test(cfg=cfg2, model=predictor.model, evaluators=[evaluator])

#inference_on_dataset()

#print(res)
