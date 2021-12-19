from torch._C import device
import detectron2
import numpy as np
import os
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets import register_coco_panoptic
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.engine import DefaultTrainer, launch
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
import sys

import torch

antall = torch.cuda.device_count()

print(antall)

for i in range(antall):
    print(torch.cuda.get_device_name(i))

register_coco_panoptic_separated(name="ffi_train", sem_seg_root="dataset_train/panoptic_stuff_train", metadata={}, image_root="dataset_train/images", panoptic_root="dataset_train/panoptic_train" , panoptic_json="dataset_train/annotations/panoptic_train.json" ,instances_json="dataset_train/annotations/instances_train.json")
register_coco_panoptic_separated(name="ffi_test", sem_seg_root="dataset_test/panoptic_stuff_test", metadata={}, image_root="dataset_test/images", panoptic_root="dataset_test/panoptic_test" , panoptic_json="dataset_test/annotations/panoptic_test.json" ,instances_json="dataset_test/annotations/instances_test.json")

MetadataCatalog.get("ffi_test_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'CameraEdge','Bush', 'Grass',  'Gravel', ])
MetadataCatalog.get("ffi_test_separated").set(stuff_colors=[[255,255,255], [15,171,255], [191,140,0], [46,153,0], [70,70,70], [232,227,81], [64,255,38], [180,180,180], ])
MetadataCatalog.get("ffi_test_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 2: 4, 5: 5, 10: 6, 11: 7})

MetadataCatalog.get("ffi_test_separated").set(thing_classes=['Vehicle','Person', 'Puddle', 'Large_stone', 'Tree', 'Building'])
MetadataCatalog.get("ffi_test_separated").set(thing_colors=[[150,0,191],[255,38,38], [255,179,0], [200,200,200], [118,255,99], [255,20,20]])
MetadataCatalog.get("ffi_test_separated").set(thing_dataset_id_to_contiguous_id={3: 0, 4: 1, 6: 2, 9: 3, 12: 4, 13: 5})

MetadataCatalog.get("ffi_train_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'CameraEdge','Bush', 'Grass',  'Gravel', ])
MetadataCatalog.get("ffi_train_separated").set(stuff_colors=[[255,255,255], [15,171,255], [191,140,0], [46,153,0], [70,70,70], [232,227,81], [64,255,38], [180,180,180], ])
MetadataCatalog.get("ffi_train_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 2: 4, 5: 5, 10: 6, 11: 7})

MetadataCatalog.get("ffi_train_separated").set(thing_classes=['Vehicle','Person', 'Puddle', 'Large_stone', 'Tree', 'Building'])
MetadataCatalog.get("ffi_train_separated").set(thing_colors=[[150,0,191],[255,38,38], [255,179,0], [200,200,200], [118,255,99], [255,20,20]])


#print(torch.cuda.memory_summary(device=None, abbreviated=False))
def train():
    cfg = get_cfg()
    #cfg.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    cfg.merge_from_file("panopticConfig.yaml")
    cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])


    cfg.DATASETS.TRAIN = ("ffi_train_separated")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8

    cfg.INPUT.MASK_FORMAT = "bitmask"
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    #DatasetCatalog.get('coco_2017_train_panoptic')
    #print(cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    print(cfg)

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    return (trainer.train())
#metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
#print(m_train2017etadata)



#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
#predictor = DefaultPredictor(cfg)


#res = evaluator.evaluate()
#print(res)




def validate():
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
                frame, panoptic_seg.to('cuda'), segments_info
            )
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

    cv2.imwrite("stille.png", vis_frame)

    evaluator_train = COCOPanopticEvaluator(dataset_name="ffi_train_separated")
    val_loader = build_detection_test_loader(cfg2, "ffi_train_separated")
    print(inference_on_dataset(predictor.model, val_loader, evaluator_train))


if __name__ == "__main__":
    print("for lunsj")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    print("port", port)
    launch(
        train,
        8,
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:{}".format(port),
    )
    print("etter lunsj")

    validate()

#evaluator_train = COCOPanopticEvaluator(dataset_name="ffi_train_separated")
#val_loader = build_detection_test_loader(cfg2, "ffi_train_separated")
#print(inference_on_dataset(predictor.model, val_loader, evaluator_train))




#evaluator_test = COCOPanopticEvaluator(dataset_name="ffi_test_separated")
#val_loader = build_detection_test_loader(cfg2, "ffi_test_separated")
#print(inference_on_dataset(predictor.model, val_loader, evaluator_test))


#res = trainer.test(cfg=cfg2, model=predictor.model, evaluators=[evaluator])

#inference_on_dataset()

#print(res)
