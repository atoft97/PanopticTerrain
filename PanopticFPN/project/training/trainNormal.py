
"""
A main training script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_panoptic_separated
import sys
import time

from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

import json
from csv import writer
from csv import DictWriter

def write_to_csv(new_data, fileName):
    print("skriver", new_data)
    with open(fileName, 'a') as f_object:
        writer_object = DictWriter(f_object, fieldnames=['PQ', 'SQ', 'RQ', 'PQ_th', 'SQ_th', 'RQ_th', 'PQ_st', 'SQ_st', 'RQ_st'])
        #writer_object = writer(f_object)
        writer_object.writerow(new_data)
        f_object.close()

timeStamp = str(time.time())

testName = "pq/pq_test"+ timeStamp + ".csv"
trainName = "pq/pq_train"+ timeStamp + ".csv"

write_to_csv({'PQ': 'PQ', 'SQ': 'SQ', 'RQ': 'RQ', 'PQ_th': 'PQ_th', 'SQ_th': 'SQ_th', 'RQ_th': 'RQ_th', 'PQ_st':'PQ_st', 'SQ_st':'SQ_st', 'RQ_st':'RQ_st'}, testName)
write_to_csv({'PQ': 'PQ', 'SQ': 'SQ', 'RQ': 'RQ', 'PQ_th': 'PQ_th', 'SQ_th': 'SQ_th', 'RQ_th': 'RQ_th', 'PQ_st':'PQ_st', 'SQ_st':'SQ_st', 'RQ_st':'RQ_st'}, trainName)

#PQfileTest = open("pq/pq_test"+ timeStamp + ".txt","a")
#PQfileTrain = open("pq/pq_train"+ timeStamp + ".txt","a")

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

def build_evaluator(cfg, dataset_name, output_folder="eval_output"):
    evaluator_list = []
    evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        print(results)
        results_test = list(results.values())[0]
        print("test:", results_test['PQ'])
        write_to_csv(results_test, testName)
        #PQfileTest.write(str(results_test['PQ']))
        #PQfileTest.write(str(results_test['PQ']))
        #PQfileTest.write("\n")

        dataset_name_train = cfg.DATASETS.TRAIN
        data_loader_train = cls.build_test_loader(cfg, dataset_name_train)
        evaluator_train = COCOPanopticEvaluator(dataset_name_train)
        results_train = inference_on_dataset(model, data_loader_train, evaluator_train)
        results_train = list(results_train.values())[0]
        print("train:", results_train['PQ'])
        write_to_csv(results_train, trainName)
        #PQfileTrain.write(str(results_train['PQ']))
        #PQfileTrain.write("\n")


        return results




def createCfg():
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    #cfg.merge_from_file("panopticConfig.yaml")
    #cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])
    cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
    cfg.MODEL.WEIGHTS = ''

    cfg.DATASETS.TRAIN = ("ffi_train_separated")
    cfg.DATASETS.TEST = ("ffi_val_separated", )
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.002

    cfg.SOLVER.MAX_ITER = 145*20
    cfg.SOLVER.STEPS = []
    cfg.TEST.EVAL_PERIOD = 145

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.OUTPUT_DIR = "./output50Normal"


    return(cfg)


def main():
    cfg = createCfg()

    '''
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    '''
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()

    #trainer = DefaultTrainer(cfg) 
    #trainer.resume_or_load(resume=False)
    #return (trainer.train())


if __name__ == "__main__":
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    #args = default_argument_parser().parse_args()
    #print("Command Line Args:", args)
    launch(
        main,
        1,
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:{}".format(port),
    )