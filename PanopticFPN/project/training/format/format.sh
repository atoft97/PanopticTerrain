#!/bin/bash
cd ..
mkdir -p dataset_train/annotations
mkdir -p dataset_train/images

mkdir -p dataset_test/annotations
mkdir -p dataset_test/images

mkdir -p dataset_validation/annotations
mkdir -p dataset_validation/images

cd format

python3 detection2panopticCOCO.py --input_json_file annotations/instances_Train.json --output_json_file annotations/panoptic_train.json --categories_json_file kategorier.json --segmentations_folder panoptic_train
python3 panopticToInstances.py --input_json_file annotations/panoptic_train.json --categories_json_file kategorier.json --things_only --output_json_file annotations/instances_train.json --segmentations_folder panoptic_train

python3 detection2panopticCOCO.py --input_json_file annotations/instances_Test.json --output_json_file annotations/panoptic_test.json --categories_json_file kategorier.json --segmentations_folder panoptic_test
python3 panopticToInstances.py --input_json_file annotations/panoptic_test.json --categories_json_file kategorier.json --things_only --output_json_file annotations/instances_test.json --segmentations_folder panoptic_test

python3 detection2panopticCOCO.py --input_json_file annotations/instances_Validation.json --output_json_file annotations/panoptic_validation.json --categories_json_file kategorier.json --segmentations_folder panoptic_validation
python3 panopticToInstances.py --input_json_file annotations/panoptic_validation.json --categories_json_file kategorier.json --things_only --output_json_file annotations/instances_validation.json --segmentations_folder panoptic_validation

python3 prepare_panoptic_fpn.py

cp annotations/panoptic_train.json /lhome/asbjotof/work/project/training/dataset_train/annotations
cp -r panoptic_stuff_train /lhome/asbjotof/work/project/training/dataset_train/
cp -r panoptic_train /lhome/asbjotof/work/project/training/dataset_train/

cp annotations/panoptic_test.json /lhome/asbjotof/work/project/training/dataset_test/annotations
#cp annotations/instances_train.json /lhome/asbjotof/work/training/dataset/annotations
cp -r panoptic_stuff_test /lhome/asbjotof/work/project/training/dataset_test/
cp -r panoptic_test /lhome/asbjotof/work/project/training/dataset_test/

cp annotations/panoptic_validation.json /lhome/asbjotof/work/project/training/dataset_validation/annotations
#cp annotations/instances_train.json /lhome/asbjotof/work/training/dataset/annotations
cp -r panoptic_stuff_validation /lhome/asbjotof/work/project/training/dataset_validation/
cp -r panoptic_validation /lhome/asbjotof/work/project/training/dataset_validation/


python3 png2jpg.py