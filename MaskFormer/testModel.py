
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
import copy
import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
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
from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_panoptic
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

from mask_former import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_mask_former_config,
)


from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor

antall = torch.cuda.device_count()
print(antall)

for i in range(antall):
    print(torch.cuda.get_device_name(i))

def write_to_csv(new_data, fileName):
    print("skriver", new_data)
    with open(fileName, 'a') as f_object:
        writer_object = DictWriter(f_object, fieldnames=['PQ', 'SQ', 'RQ', 'PQ_th', 'SQ_th', 'RQ_th', 'PQ_st', 'SQ_st', 'RQ_st'])
        #writer_object = writer(f_object)
        writer_object.writerow(new_data)
        f_object.close()

timeStamp = str(time.time())

testName = "pq/MaskLLongTest"+ timeStamp + ".csv"
trainName = "pq/MaskLLongTrain"+ timeStamp + ".csv"

#write_to_csv({'PQ': 'PQ', 'SQ': 'SQ', 'RQ': 'RQ', 'PQ_th': 'PQ_th', 'SQ_th': 'SQ_th', 'RQ_th': 'RQ_th', 'PQ_st':'PQ_st', 'SQ_st':'SQ_st', 'RQ_st':'RQ_st'}, testName)
#write_to_csv({'PQ': 'PQ', 'SQ': 'SQ', 'RQ': 'RQ', 'PQ_th': 'PQ_th', 'SQ_th': 'SQ_th', 'RQ_th': 'RQ_th', 'PQ_st':'PQ_st', 'SQ_st':'SQ_st', 'RQ_st':'RQ_st'}, trainName)

#PQfileTest = open("pq/pq_test"+ timeStamp + ".txt","a")
#PQfileTrain = open("pq/pq_train"+ timeStamp + ".txt","a")

register_coco_panoptic(name="ffi_train",metadata={"stuff_dataset_id_to_contiguous_id":{8: 1, 7: 2, 1: 4, 2: 5, 5: 6, 10: 9, 11: 10},"thing_dataset_id_to_contiguous_id":{4: 0, 3: 3, 6: 7, 9: 8, 12: 11, 13: 12}} , image_root="dataset_train/images", panoptic_root="dataset_train/panoptic_train" , panoptic_json="dataset_train/annotations/panoptic_train.json" ,instances_json="dataset_train/annotations/instances_train.json")
#register_coco_panoptic(name="ffi_val", metadata={"stuff_dataset_id_to_contiguous_id":{8: 1, 7: 2, 1: 4, 2: 5, 5: 6, 10: 9, 11: 10},"thing_dataset_id_to_contiguous_id":{4: 0, 3: 3, 6: 7, 9: 8, 12: 11, 13: 12}}, image_root="dataset_validation/images", panoptic_root="dataset_validation/panoptic_validation" , panoptic_json="dataset_validation/annotations/panoptic_validation.json" ,instances_json="dataset_validation/annotations/instances_validation.json")
register_coco_panoptic(name="ffi_test", metadata={"stuff_dataset_id_to_contiguous_id":{8: 1, 7: 2, 1: 4, 2: 5, 5: 6, 10: 9, 11: 10},"thing_dataset_id_to_contiguous_id":{4: 0, 3: 3, 6: 7, 9: 8, 12: 11, 13: 12}}, image_root="dataset_test/images", panoptic_root="dataset_test/panoptic_test" ,panoptic_json="dataset_test/annotations/panoptic_test.json", instances_json="dataset_test/annotations/instances_test.json")


MetadataCatalog.get("ffi_train").set(stuff_classes=['Person', 'Sky', 'Dirtroad', 'Vehicle', 'Forrest', 'CameraEdge', 'Bush', 'Puddle', 'Large_stone', 'Grass', 'Gravel', 'Tree', 'Building'])
MetadataCatalog.get("ffi_train").set(stuff_colors=[[255, 38, 38], [15, 171, 255], [191, 140, 0], [150, 0, 191], [46, 153, 0], [70, 70, 70], [232, 227, 81], [255, 179, 0], [200, 200, 200], [64, 255, 38], [180, 180, 180], [118, 255, 99], [255, 20, 20]])
MetadataCatalog.get("ffi_train").set(stuff_dataset_id_to_contiguous_id={8: 1, 7: 2, 1: 4, 2: 5, 5: 6, 10: 9, 11: 10})


MetadataCatalog.get("ffi_train").set(thing_classes=['Person', 'Sky', 'Dirtroad', 'Vehicle', 'Forrest', 'CameraEdge', 'Bush', 'Puddle', 'Large_stone', 'Grass', 'Gravel', 'Tree', 'Building'])
MetadataCatalog.get("ffi_train").set(thing_colors=[[255, 38, 38], [15, 171, 255], [191, 140, 0], [150, 0, 191], [46, 153, 0], [70, 70, 70], [232, 227, 81], [255, 179, 0], [200, 200, 200], [64, 255, 38], [180, 180, 180], [118, 255, 99], [255, 20, 20]])
MetadataCatalog.get("ffrom detectron2.engine import DefaultPredictorfi_train").set(thing_dataset_id_to_contiguous_id={4: 0, 3: 3, 6: 7, 9: 8, 12: 11, 13: 12})


MetadataCatalog.get("ffi_test").set(stuff_classes=['Person', 'Sky', 'Dirtroad', 'Vehicle', 'Forrest', 'CameraEdge', 'Bush', 'Puddle', 'Large_stone', 'Grass', 'Gravel', 'Tree', 'Building'])
MetadataCatalog.get("ffi_test").set(stuff_colors=[[255, 38, 38], [15, 171, 255], [191, 140, 0], [150, 0, 191], [46, 153, 0], [70, 70, 70], [232, 227, 81], [255, 179, 0], [200, 200, 200], [64, 255, 38], [180, 180, 180], [118, 255, 99], [255, 20, 20]])
MetadataCatalog.get("ffi_test").set(stuff_dataset_id_to_contiguous_id={8: 1, 7: 2, 1: 4, 2: 5, 5: 6, 10: 9, 11: 10})


MetadataCatalog.get("ffi_test").set(thing_classes=['Person', 'Sky', 'Dirtroad', 'Vehicle', 'Forrest', 'CameraEdge', 'Bush', 'Puddle', 'Large_stone', 'Grass', 'Gravel', 'Tree', 'Building'])
MetadataCatalog.get("ffi_test").set(thing_colors=[[255, 38, 38], [15, 171, 255], [191, 140, 0], [150, 0, 191], [46, 153, 0], [70, 70, 70], [232, 227, 81], [255, 179, 0], [200, 200, 200], [64, 255, 38], [180, 180, 180], [118, 255, 99], [255, 20, 20]])
MetadataCatalog.get("ffi_test").set(thing_dataset_id_to_contiguous_id={4: 0, 3: 3, 6: 7, 9: 8, 12: 11, 13: 12})


def build_evaluator(cfg, dataset_name, output_folder="eval_output"):
    evaluator_list = []
    evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # DETR-style dataset mapper for COCO panoptic segmentation
        elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
            mapper = DETRPanopticDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)
    
   

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        print(res)
        return res

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)
    
    '''
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
        
        dataset_name = cfg.DATASETS.TEST[0]
        #for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        # When evaluators are passed in as arguments,
        # implicitly assume that evaluators can be created before data_loader.
        #if evaluators is not None:
        #    evaluator = evaluators[idx]
        evaluator = evaluators[0]
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

        dataset_name_train = cfg.DATASETS.TRAIN
        print(dataset_name_train)
        data_loader_train = cls.build_test_loader(cfg, dataset_name_train)
        print(data_loader_train)
        evaluator_train = COCOPanopticEvaluator(dataset_name_train)
        print(evaluator_train)
        results_train = inference_on_dataset(model, data_loader_train, evaluator_train)
        print(results_train)
        results_train = list(results_train.values())[0]
        print("train:", results_train['PQ'])
        write_to_csv(results_train, trainName)

        dataset_name_test = 'ffi_val'
        print(dataset_name_test)
        data_loader_test = cls.build_test_loader(cfg, dataset_name_test)
        print(data_loader_test)
        evaluator_test = COCOPanopticEvaluator(dataset_name_test)
        print(evaluator_test)
        results_test = inference_on_dataset(model, data_loader_test, evaluator_test)
        print(results_test)
        results_test = list(results_test.values())[0]
        print("train:", results_test['PQ'])
        write_to_csv(results_test, testName)


        return results
    '''



def createCfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file("configs/coco-panoptic/swin/maskformer_panoptic_swin_large_IN21k_384_bs64_554k.yaml")
    #cfg.merge_from_file("panopticConfig.yaml")
    #cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])
    cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
    cfg.MODEL.WEIGHTS = 'models/00002.pth'

    cfg.DATASETS.TRAIN = ("ffi_train", )
    cfg.DATASETS.TEST = ("ffi_test", )
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.002

    cfg.SOLVER.MAX_ITER = 145*20
    cfg.SOLVER.STEPS = []
    cfg.TEST.EVAL_PERIOD = 145

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 13

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.OUTPUT_DIR = "./outputMaskL"

    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")

    return(cfg)


args = default_argument_parser().parse_args()
cfg = createCfg(args)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
#trainer.train()

predictor = DefaultPredictor(cfg)
evaluator_test = COCOPanopticEvaluator(dataset_name="ffi_test")
val_loader = build_detection_test_loader(cfg, dataset_name="ffi_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator_test))




