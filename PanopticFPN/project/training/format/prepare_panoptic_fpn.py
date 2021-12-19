#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

#print(COCO_CATEGORIES)

COCO_CATEGORIES = [
{"supercategory": "Person",     "color": [255,38,38],   "isthing": 1, "id": 4,  "name": "Person"},
{"supercategory": "Sky",        "color": [15,171,255],  "isthing": 0, "id": 8,  "name": "Sky"},
{"supercategory": "Dirtroad",   "color": [191,140,0],   "isthing": 0, "id": 7,  "name": "Dirtroad"},
{"supercategory": "Vehicle",    "color": [150,0,191],   "isthing": 1, "id": 3,  "name": "Vehicle"},
{"supercategory": "Forrest",    "color": [46,153,0],    "isthing": 0, "id": 1,  "name": "Forrest"},
{"supercategory": "CameraEdge", "color": [70,70,70],    "isthing": 0, "id": 2, "name": "CameraEdge"},
{"supercategory": "Bush",       "color": [232,227,81],  "isthing": 0, "id": 5,  "name": "Bush"},
{"supercategory": "Puddle",     "color": [255,179,0],   "isthing": 1, "id": 6,  "name": "Puddle"},
{"supercategory": "Large_stone","color": [200,200,200], "isthing": 1, "id": 9, "name": "Large_stone"},
{"supercategory": "Grass",      "color": [64,255,38],   "isthing": 0, "id": 10,  "name": "Grass"},
{"supercategory": "Gravel",     "color": [180,180,180], "isthing": 0, "id": 11,  "name": "Gravel"},
{"supercategory": "Tree",       "color": [118,255,99],  "isthing": 1, "id": 12,  "name": "Tree"},
{"supercategory": "Building",   "color": [255,20,20],   "isthing": 1, "id": 13,  "name": "Building"}
]
def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    print(id_map)
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    print(new_cat_id)
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.

    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]
    thing_ids = [k["id"] for k in categories if k["isthing"] == 1]
    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(stuff_ids) <= 254
    for i, stuff_id in enumerate(stuff_ids):
        id_map[stuff_id] = i + 1
    for thing_id in thing_ids:
        id_map[thing_id] = 0
    id_map[0] = 255
    
    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    print("asdasdasd")
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
    for s in ["train", "test", "validation"]:
        separate_coco_semantic_from_panoptic(
            os.path.join("annotations/panoptic_"+s+".json"),
            os.path.join("panoptic_"+s),
            os.path.join("panoptic_stuff_"+s),
            COCO_CATEGORIES,
        )
        #separate_coco_semantic_from_panoptic(
        #    os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
        #    os.path.join(dataset_dir, "annotations/panoptic_{}".format(s)),
        #    os.path.join(dataset_dir, "panoptic_stuff_{}".format(s)),
        #    COCO_CATEGORIES,
        #)

    # Prepare val2017_100 for quick testing:

    dest_dir = os.path.join(dataset_dir, "annotations/")
    URL_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"
    download(URL_PREFIX + "annotations/coco/panoptic_val2017_100.json", dest_dir)
    with open(os.path.join(dest_dir, "panoptic_val2017_100.json")) as f:
        obj = json.load(f)

    def link_val100(dir_full, dir_100):
        print("Creating " + dir_100 + " ...")
        os.makedirs(dir_100, exist_ok=True)
        for img in obj["images"]:
            basename = os.path.splitext(img["file_name"])[0]
            src = os.path.join(dir_full, basename + ".png")
            dst = os.path.join(dir_100, basename + ".png")
            src = os.path.relpath(src, start=dir_100)
            os.symlink(src, dst)

    #link_val100(
    #    os.path.join(dataset_dir, "panoptic_val2017"),
    #    os.path.join(dataset_dir, "panoptic_val2017_100"),
    #)

    #link_val100(
    #    os.path.join(dataset_dir, "panoptic_stuff_val2017"),
    #    os.path.join(dataset_dir, "panoptic_stuff_val2017_100"),
    #)
