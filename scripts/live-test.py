import os
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np
from datetime import datetime

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations
from scripts.fall_down_detect import *
from testing.Test import Test
from datasets.COCO import COCODataset
from training.COCO import COCOTrain


def main( filename, foldername,hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
         single_person, disable_tracking, max_batch_size, disable_vidgear, save_video, video_format,
         video_framerate, device):

    # torch device
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:1')
        else:
            device = torch.device('cpu')

    print(device)

    #print("\nStarting experiment `%s` @ %s\n" % ( datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    #lr_decay = not disable_lr_decay
    #use_tensorboard = not disable_tensorboard_log
    #flip_test_images = not disable_flip_test_images
    #image_resolution = ast.literal_eval(image_resolution)
    #lr_decay_steps = ast.literal_eval(lr_decay_steps)


    image_resolution=(384, 288)
    coco_root_path="/database/dataset/MS-COCO2017"
    
    '''
    ds_train = COCODataset(
        root_path=coco_root_path, data_version="train2017", is_train=True, use_gt_bboxes=True, bbox_path="",
        image_width=image_resolution[1], image_height=image_resolution[0], color_rgb=True,
    )
    '''

    ds_test = COCODataset(
        root_path=coco_root_path, data_version="val2017", is_train=True, use_gt_bboxes=True, bbox_path="",
        image_width=image_resolution[1], image_height=image_resolution[0], color_rgb=True,
    )


    test = Test(ds_test, checkpoint_path="./weights/pose_hrnet_w48_384x288.pth")
    test.run()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--foldername", help="open the specified folder (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                                     "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
