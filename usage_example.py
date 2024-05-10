import os
from typing import Dict

import cv2
import pandas as pd
import torch
from torch import nn

from Models.models import Uni_modal_ER_model, Multi_modal_ER_model
from feature_extraction.pytorch_based.face_recognition_utils import recognize_one_face_bbox, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace_mobileNet
from feature_extraction.pytorch_based.pose_recognition_utils import crop_frame_to_pose
from simpleHRNet.SimpleHRNet import SimpleHRNet


def get_ER_model(config:Dict[str, str])->nn.Module:
    """
    Initializes the required model, loads weights for it, and returns the model
    :param config: Dict[str, str]
        The configuration dictionary with the following keys:
        1. model_type -- The type of the model you want to use. The supported versions are:
            - "Static_facial_ER": Static facial engagement recognition model (recognition of engagement from facial expressions)
            - "Static_kinesics_ER": Static kinesics engagement recognition model (recognition of engagement from body movements)
            - "Dynamic_uni_modal_facial_ER": Dynamic facial engagement recognition model (recognition of engagement from
                the sequence of facial expressions).
            - "Dynamic_uni_modal_kinesics_ER": Dynamic kinesics engagement recognition model (recognition of engagement from
                the sequence of body movements).
            - "Dynamic_multi_modal_kinesics_facial_ER": Dynamic multimodal engagement recognition model (recognition of
                engagement from the sequence of facial expressions and body movements).
            - "Dynamic_multi_modal_affective_kinesics_ER": Dynamic multimodal engagement recognition model (recognition of
                engagement from the sequence of body movements and affective features).
            - "Dynamic_multi_modal_affective_facial_ER": Dynamic multimodal engagement recognition model (recognition of
                engagement from the sequence of facial expressions and affective features).
            - "Dynamic_multi_modal_all_ER": Dynamic multimodal engagement recognition model (recognition of engagement from
                the sequence of facial expressions, body movements, and affective features).

        2. static_model_paths -- The paths to the static model that will be used either as main model (if you have chosen
                a static model) or as a feature extractor (if you have chosen a dynamic model). Note that the value
                of this key should be a list of paths as for multi-modal models, you have several static models (e. g. facial and kinesics static models)
        3. dynamic_model_path -- The path to the dynamic model that will be used as the main model. This key is ignored
                if you have chosen a static model.
        4. hrnet_weights_path -- The path to the weights of the HRNet model. This key is ignored if you have not chosen
        a model that uses HRNet (any kinesics model).
    :return: nn.Module

    """
    if "Static" in config["model_type"] or "uni_modal" in config["model_type"]:
        model = Uni_modal_ER_model(config)
    elif "multi_modal" in config["model_type"]:
        model = Multi_modal_ER_model(config)
    else:
        raise ValueError("Invalid model type. The model type should be one of the following: "
                         "'Static_facial_ER', 'Static_kinesics_ER', 'Dynamic_uni_modal_facial_ER', "
                         "'Dynamic_uni_modal_kinesics_ER', 'Dynamic_multi_modal_kinesics_facial_ER', "
                         "'Dynamic_multi_modal_affective_kinesics_ER', 'Dynamic_multi_modal_affective_facial_ER', "
                         "or 'Dynamic_multi_modal_all_ER'.")
    model.eval()
    return model



def process_video_bi_modal(path_to_video:str, face_fetector, pose_detector, ER_model) -> pd.DataFrame:
    """
    Process the video and return the engagement recognition results with outputing as print simultaneously.
    :param path_to_video: str
        The path to the videofile
    :param face_fetector: Callable
        The face detector. We use retinaFace detector in his work.
    :param pose_detector: Callable
        The pose detector. We use HRNet pose detector in this work.
    :param ER_model: nn.Module
        The engagement recognition model. Can be static, uni-modal dynamic or multi-modal dynamic. Should be initialized
        with the get_ER_model function before.
    :return: pd.DataFrame
    """
    metadata = pd.DataFrame(columns=["start_time", "end_time", "engagement_state"])
    # load video file
    video = cv2.VideoCapture(path_to_video)
    # get FPS
    FPS = video.get(cv2.CAP_PROP_FPS)
    FPS_in_seconds = 1. / FPS
    # calculate every which frame should be taken (we downgrade everything to 5 FPS)
    every_n_frame = int(round(FPS / 5))
    # go through all frames
    counter = 0
    last_face = torch.zeros(1, 3, 224, 224)
    last_pose = torch.zeros(1, 3, 256, 256)
    # params for the model
    faces = []
    poses = []
    acc_steps = 0
    need_steps = int(8*5) # 8 seconds with 5 FPS
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter % every_n_frame == 0:
                # calculate timestamp
                timestamp = counter * FPS_in_seconds
                # round it to 2 digits to make it readable
                timestamp = round(timestamp, 2)
                # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # recognize the face
                face_bbox = recognize_one_face_bbox(frame, face_fetector)
                # recognize the pose
                pose_bbox = pose_detector(frame)
                # if face is not recognized, use the last recognized face
                if face_bbox is None:
                    face = last_face
                else:
                    face = extract_face_according_bbox(frame, face_bbox)
                # if pose is not recognized, use the last recognized pose
                if pose_bbox is None:
                    pose = last_pose
                else:
                    pose = crop_frame_to_pose(frame, pose_bbox, return_bbox=False)
                # make prediction if we have enough steps
                if acc_steps < need_steps:
                    faces.append(face)
                    poses.append(pose)
                    acc_steps += 1
                else:
                    faces = torch.stack(faces)
                    poses = torch.stack(poses)
                    # send to model. Order is important (see Multi_modal_ER_model class, forward function).
                    input = []
                    for static_model_types in ER_model.static_model_types:
                        if static_model_types in ["facial", "affective"]:
                            input.append(faces)
                        elif static_model_types in ["kinesics"]:
                            input.append(poses)
                    # make prediction
                    prediction = ER_model(input)
                    # output the result
                    row = {
                        "start_time": [timestamp - 8.],
                        "end_time": [timestamp],
                        "engagement_state": [prediction]
                    }
                    metadata = pd.concat([metadata, pd.DataFrame(row)], ignore_index=True)
                    # reset all variables
                    faces = []
                    poses = []
                    acc_steps = 0

                # update last face and pose
                last_face = face
                last_pose = pose

            counter += 1
        else:
            break
    video.release()
    return metadata




def main():
    config = {
        "model_type": "Dynamic_multi_modal_affective_kinesics_ER",
        "static_model_paths": [
            "/work/home/dsu/PhD/Model_weights/affective_static_efficientNet_b1.pth",
            "/work/home/dsu/PhD/Model_weights/kinesics_engagement_static_hrnet.pth"
        ],
        "dynamic_model_path": "/work/home/dsu/PhD/Model_weights/Engagement/two_modal_affective_kinesics.pth",
        "hrnet_weights_path": "/work/home/dsu/PhD/scripts/simple-HRNet-master/pose_hrnet_w48_384x288.pth",
        "path_to_project": os.path.dirname(os.path.abspath(__file__))
    }
    path_to_video = "/work/home/dsu/Datasets/video83.mp4"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ER_model = get_ER_model(config)
    face_detector = load_and_prepare_detector_retinaFace_mobileNet(device="cpu")
    pose_detector = SimpleHRNet(c=48, nof_joints=17, multiperson=True,
                               yolo_version = 'v3',
                               yolo_model_def=os.path.join(config["path_to_project"],"models_/detectors/yolo/config/yolov3.cfg"),
                               yolo_class_path=os.path.join(config["path_to_project"],"models_/detectors/yolo/data/coco.names"),
                               yolo_weights_path=os.path.join(config["path_to_project"],"models_/detectors/yolo/weights/yolov3.weights"),
                               checkpoint_path=os.path.join(config["path_to_project"], "pose_hrnet_w48_384x288.pth"),
                               return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1, device=torch.device(device))
    result = process_video_bi_modal(path_to_video, face_detector, pose_detector, ER_model)
    print(result)




if __name__ == "__main__":
    main()





