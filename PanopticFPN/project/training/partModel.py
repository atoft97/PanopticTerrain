from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
import torch
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from  detectron2.structures.image_list import ImageList
from detectron2.data.datasets import register_coco_panoptic_separated
import numpy as np

cfg = get_cfg()

cfg.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
#cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])

register_coco_panoptic_separated(name="ffi_test", sem_seg_root="dataset_test/panoptic_stuff_test", metadata={}, image_root="dataset_test/images", panoptic_root="dataset_test/panoptic_test" , panoptic_json="dataset_test/annotations/panoptic_test.json" ,instances_json="dataset_test/annotations/instances_test.json")


cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
cfg.MODEL.WEIGHTS = "output/model_final.pth"
#cfg2.MODEL.WEIGHTS = "detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl"
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
#cfg.DATASETS.TEST = ("ffi_test_separated", )
cfg.DATASETS.TEST = ("ffi_test_separated", )


cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8
cfg.INPUT.MASK_FORMAT = "bitmask"

MetadataCatalog.get("ffi_test_separated").set(stuff_classes=['things', 'Sky', 'Dirtroad', 'Forrest', 'Bush', 'Grass', 'Tall_grass', 'Gravel'])
MetadataCatalog.get("ffi_test_separated").set(stuff_colors=[[255,255,255], [15,171,255], [191,140,0], [46,153,0], [232,227,81], [64,255,38], [255,251,38], [180,180,180]])
MetadataCatalog.get("ffi_test_separated").set(stuff_dataset_id_to_contiguous_id={255:0,  8: 1, 7: 2, 1: 3, 5: 4, 10: 5, 11: 6, 12: 7})

MetadataCatalog.get("ffi_test_separated").set(thing_classes=['CameraEdge','Vehicle','Person',   'Puddle', 'Large_stone', ])
MetadataCatalog.get("ffi_test_separated").set(thing_colors=[[70,70,70], [150,0,191],[255,38,38],   [255,179,0], [200,200,200], ])



model = build_model(cfg)
model.eval()

checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

frame = read_image("dataset_train/images/stille_frame00001.jpg", format="BGR")


print(model)
print(cfg.INPUT)



aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
'''
height, width = frame.shape[:2]
image = aug.get_transform(frame).apply_image(frame)
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
inputs = {"image": image, "height": height, "width": width}

print(image)
image = image[None, :]

device = torch.device('cuda')
image =  image.to(device)
features = model.backbone([inputs])
print("features")
print(features)
'''
device = torch.device("cuda")
def predictImage(frame):

    with torch.no_grad():
        height, width = frame.shape[:2]
        print(frame.shape)
        image = aug.get_transform(frame).apply_image(frame)
        print(image.shape)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        
    #	print(inputs)
        image = image.to(device) 
        images = ImageList.from_tensors([image], model.backbone.size_divisibility)
        #predictions = model([inputs])
        #print(type(images))
        #print(model.backbone.forward)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features)
        instances, _ = model.roi_heads(images, features, proposals)

        segmetnts, _ = model.sem_seg_head(features) 
        segmetnts = segmetnts.cpu().detach().numpy()

        segmetnts = np.squeeze(segmetnts, axis=0)
        print(segmetnts[0]*10)
        print("form", segmetnts[0].shape)
        for i in range(8):
            cv2.imwrite("partial/yo"+str(i)+".png", segmetnts[i]*10)

        #print(segmetnts)
        #print(instances)
        #print(images.tensor.shape)
        #predictions = model.backbone(torch.randn(1, 3, 800, 799).to(device))
        #print(predictions)
        #predictions = predictions[0]

    #predictedPanoptic = predictions

    #return(predictedPanoptic)


outputs = predictImage(frame)

#predictor = DefaultPredictor(cfg)
#outputs = predictor(frame)
'''
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

panoptic_seg, segments_info = outputs["panoptic_seg"]

vis_frame = video_visualizer.draw_panoptic_seg_predictions(
            frame, panoptic_seg.to('cuda'), segments_info
        )
vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

cv2.imwrite("stille.png", vis_frame)
'''