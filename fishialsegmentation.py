import json
import cv2
import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from projects.PointRend.point_rend import add_pointrend_config
import matplotlib.pyplot as plt

def read_json(path):
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        return data
    else:
        return None

class SegmentationInference:
    def __init__(self, model_path, config, device):
        self.model_path = model_path
        
        self.cfg = get_cfg()
        add_pointrend_config(self.cfg)
        self.cfg.merge_from_file(config)
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.freeze()
        self.model = DefaultPredictor(self.cfg)

    def inference(self, img):
        outputs = self.model(img)
        return outputs
    
path_to_segmentation_model = r'Models/model_15_11_2022.pth'
path_to_segmentation_config = r'detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml'

model_segmentation = SegmentationInference(
        path_to_segmentation_model,
        config = path_to_segmentation_config,
        device = 'cpu')

image_path = 'fredfish3.jpg'
image_name = os.path.basename(image_path)
image_name = os.path.splitext(image_name)[0]
im = cv2.imread(image_path)


outputs = model_segmentation.inference(im)

for mask_id in range(len(outputs['instances'].pred_boxes.tensor)):
    x_1, y_1, x_2, y_2 = outputs['instances'].pred_boxes.tensor[mask_id].numpy()
    mask = outputs['instances'].pred_masks[mask_id].numpy()
    mask = mask[int(y_1):int(y_2), int(x_1):int(x_2)].copy()
    img = im[int(y_1):int(y_2), int(x_1):int(x_2)].copy()

    img[mask==0] = (0,0,0)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join('fish_images',image_name+str(mask_id)+'.jpg'), bbox_inches='tight') #save image directory
    plt.show()