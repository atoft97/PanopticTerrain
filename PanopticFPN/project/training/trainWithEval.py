from torch._C import device
import detectron2
import numpy as np
import os
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets import register_coco_panoptic
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, default_writers, launch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
build_evaluator
from detectron2.evaluation.panoptic_evaluation import COCOPanopticEvaluator
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.modeling import build_model
from detectron2.evaluation import inference_on_dataset

from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import read_image
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import sys
import time
import torch

import logging
import detectron2.utils.comm as comm
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
logger = logging.getLogger("detectron2")

antall = torch.cuda.device_count()
from collections import OrderedDict
print(antall)

for i in range(antall):
    print(torch.cuda.get_device_name(i))

register_coco_panoptic_separated(name="ffi_train", sem_seg_root="dataset_train/panoptic_stuff_train", metadata={}, image_root="dataset_train/images", panoptic_root="dataset_train/panoptic_train" , panoptic_json="dataset_train/annotations/panoptic_train.json" ,instances_json="dataset_train/annotations/instances_train.json")
register_coco_panoptic_separated(name="ffi_val", sem_seg_root="dataset_validation/panoptic_stuff_validation", metadata={}, image_root="dataset_validation/images", panoptic_root="dataset_validation/panoptic_validation" , panoptic_json="dataset_validation/annotations/panoptic_validation.json" ,instances_json="dataset_validation/annotations/instances_validation.json")



MetadataCatalog.get("ffi_val_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'CameraEdge','Bush', 'Grass',  'Gravel', ])
MetadataCatalog.get("ffi_val_separated").set(stuff_colors=[[255,255,255], [15,171,255], [191,140,0], [46,153,0], [70,70,70], [232,227,81], [64,255,38], [180,180,180], ])
MetadataCatalog.get("ffi_val_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 2: 4, 5: 5, 10: 6, 11: 7})

MetadataCatalog.get("ffi_val_separated").set(thing_classes=['Vehicle','Person', 'Puddle', 'Large_stone', 'Tree', 'Building'])
MetadataCatalog.get("ffi_val_separated").set(thing_colors=[[150,0,191],[255,38,38], [255,179,0], [200,200,200], [118,255,99], [255,20,20]])
MetadataCatalog.get("ffi_val_separated").set(thing_dataset_id_to_contiguous_id={3: 0, 4: 1, 6: 2, 9: 3, 12: 4, 13: 5})

MetadataCatalog.get("ffi_train_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'CameraEdge','Bush', 'Grass',  'Gravel', ])
MetadataCatalog.get("ffi_train_separated").set(stuff_colors=[[255,255,255], [15,171,255], [191,140,0], [46,153,0], [70,70,70], [232,227,81], [64,255,38], [180,180,180], ])
MetadataCatalog.get("ffi_train_separated").set(stuff_dataset_id_to_contiguous_id={255:0, 8: 1, 7: 2, 1: 3, 2: 4, 5: 5, 10: 6, 11: 7})

MetadataCatalog.get("ffi_train_separated").set(thing_classes=['Vehicle','Person', 'Puddle', 'Large_stone', 'Tree', 'Building'])
MetadataCatalog.get("ffi_train_separated").set(thing_colors=[[150,0,191],[255,38,38], [255,179,0], [200,200,200], [118,255,99], [255,20,20]])

#PQfile = open("pq.txt","w")
#os.remove("pq.txt")
timeStamp = str(time.time())

PQfile = open("pq/pq_test"+ timeStamp + ".txt","a")
PQfileTrain = open("pq/pq_train"+ timeStamp + ".txt","a")

def createCfg():
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    #cfg.merge_from_file("panopticConfig.yaml")
    cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])
    #cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])


    cfg.DATASETS.TRAIN = ("ffi_train_separated")
    cfg.DATASETS.TEST = ("ffi_val_separated", )
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 145*10
    cfg.SOLVER.STEPS = []
    cfg.TEST.EVAL_PERIOD = 145

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.OUTPUT_DIR = "./output50"

    return(cfg)

#print(torch.cuda.memory_summary(device=None, abbreviated=False))
def train(cfg, model, resume):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    print("Starting training from iteration {}".format(start_iter))


    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())

            

            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                print("iteratrion:", iteration)
                print("epoch:", (iteration+1)//145)
                print("losees:", losses)
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

    #trainer = DefaultTrainer(cfg) 
    #trainer.resume_or_load(resume=False)
    #return (trainer.train())
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


def do_test(cfg, model):
    try:
        dataset_name = cfg.DATASETS.TEST[0]
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOPanopticEvaluator(dataset_name)
        results = inference_on_dataset(model, data_loader, evaluator)

        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results)    
        
        #if len(results) == 1:
        #    results = list(results.values())[0]
        results = list(results.values())[0]
        #print(results)
        print("test:", results['PQ'])
        PQfile.write(str(results['PQ']))
        PQfile.write("\n")

        dataset_name_train = cfg.DATASETS.TRAIN
        data_loader_train = build_detection_test_loader(cfg, dataset_name_train)
        evaluator_train = COCOPanopticEvaluator(dataset_name_train)
        results_train = inference_on_dataset(model, data_loader_train, evaluator_train)

        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results)    
        
        #if len(results) == 1:
        #    results = list(results.values())[0]
        results_train = list(results_train.values())[0]
        #print(results_train)
        print("train:", results_train['PQ'])
        PQfileTrain.write(str(results_train['PQ']))
        PQfileTrain.write("\n")



        return results
    except:
        print("fail")
        return(0)


def main():
    cfg = createCfg()
    #print(cfg)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    train(cfg, model, resume=False)
    return do_test(cfg, model)


if __name__ == "__main__":
    print("for lunsj")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    print("port", port)
    launch(
        main,
        1,
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:{}".format(port),
    )
    print("etter lunsj")

    #validate()

#evaluator_train = COCOPanopticEvaluator(dataset_name="ffi_train_separated")
#val_loader = build_detection_test_loader(cfg2, "ffi_train_separated")
#print(inference_on_dataset(predictor.model, val_loader, evaluator_train))




#evaluator_test = COCOPanopticEvaluator(dataset_name="ffi_test_separated")
#val_loader = build_detection_test_loader(cfg2, "ffi_test_separated")
#print(inference_on_dataset(predictor.model, val_loader, evaluator_test))


#res = trainer.test(cfg=cfg2, model=predictor.model, evaluators=[evaluator])

#inference_on_dataset()

#print(res)
