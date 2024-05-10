import sys
import glob
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import os

import pandas as pd

from emotion_recognition.datasets.pose.hrnet_utils import cut_frame_to_pose
from simpleHRNet.SimpleHRNet import SimpleHRNet


def load_labels(path_to_labels:str)->pd.DataFrame:
    """ Load labels from the csv file and convert them to the dataframe with the following columns:
        ["filename", "timestamp", "label_value"]

    :param path_to_labels: str
        path to the csv file with labels
    :return: pd.DataFrame
        dataframe with the following columns: ["filename", "timestamp", "label_value"]
    """
    df = pd.read_csv(path_to_labels)
    df.columns = ["filename", "label_value"]
    # separate the timestamp and the filename
    df["timestamp"] = df["filename"].apply(lambda x: x.split("_")[-1])
    df["filename"] = df["filename"].apply(lambda x: x.split("_")[0])
    # reorder the columns
    df = df[["filename", "timestamp", "label_value"]]
    return df



def _extract_poses_from_one_video(path_to_video:str, detector:object, output_path:str, labels:pd.DataFrame,
                                  every_n_frame:Optional[int]=1)->pd.DataFrame:
    """ Extracts poses from one video and saves them to the output path. Also, generates the metadata for the extracted
        poses, which is returned by this function.

    :param path_to_video: str
        path to the video file
    :param detector: object
        Pose detector, HigherHrNet
    :param output_path: str
        path to the output folder, where the extracted poses will be saved
    :param labels: pd.DataFrame
        dataframe with labels with the following columns: ["filename", "timestamp", "arousal", "valence"]
    :param every_n_frame: int
        extract poses from every n-th frame
    :return: pd.DataFrame
        dataframe with the following columns: ["filename", "timestamp", "arousal", "valence"]
    """
    # metadata
    metadata = pd.DataFrame(columns=["filename", "timestamp", "arousal", "valence"])
    # load video file
    video = cv2.VideoCapture(path_to_video)
    # get FPS
    FPS = video.get(cv2.CAP_PROP_FPS)
    FPS_in_seconds = 1. / FPS
    # go through all frames
    counter=0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter% every_n_frame==0:
                # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # calculate timestamp
                timestamp = counter * FPS_in_seconds
                # round it to 2 digits to make it readable
                timestamp = round(timestamp, 2)
                # recognize the pose
                pose = cut_frame_to_pose(extractor=detector, frame=frame, return_bbox=False)
                # if not recognized, note it as NaN
                if pose is None:
                    output_filename = np.NaN
                else:
                    # otherwise save it
                    output_filename = os.path.join(output_path, path_to_video.split(os.path.sep)[-1].split(".")[0]
                                                   + f"_%s.png"%(str(timestamp).replace(".", "_")))
                    # save extracted pose
                    Image.fromarray(pose).save(output_filename)
                # add the metadata
                timestep_labels = labels[labels["timestamp"]==timestamp][["arousal", "valence"]].values
                if len(timestep_labels)==0:
                    arousal, valence = np.NaN, np.NaN
                else:
                    arousal, valence = timestep_labels[0]
                metadata = pd.concat([metadata,
                                      pd.DataFrame.from_records([{
                                            "filename": output_filename,
                                            "timestamp": timestamp,
                                            "arousal": arousal,
                                            "valence": valence}
                                      ])
                                      ], ignore_index=True)
            # increment counter
            counter += 1
        else:
            break
    return metadata





def extract_poses_with_filenames_in_dataframe(df_with_filenames:pd.DataFrame, path_to_data:str, output_path:str,
                                              every_n_frames:Optional[int]=1)->pd.DataFrame:
    """ Extracts poses from the videos in the dataframe (and available through the path_to_data) and saves them to
    the output path. Also, generates the metadata for all extracted poses, which is being returned by this function.

    :param df_with_filenames:
    :param path_to_data:
    :param output_path:
    :param every_n_frames:
    :return:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create the metadata dataframe to store the paths to every frame and corresponding labels
    metadata_all=pd.DataFrame(columns=["filename", "timestamp", "arousal", "valence"])
    detector = SimpleHRNet(c=32, nof_joints=17, checkpoint_path="/work/home/dsu/simpleHigherHRNet/pose_higher_hrnet_w32_512.pth",
                                 return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1, device="cuda")
    # search for unique video filenames in provided dataframe
    unique_video_filenames = df_with_filenames["filename"].unique()
    # go through different videos
    for video_filename in unique_video_filenames:
        # create variables needed to run _extract_poses_from_one_video() function
        video_filename = os.path.basename(glob.glob(os.path.join(path_to_data, video_filename + "*"))[0])
        path_to_video = os.path.join(path_to_data, video_filename)
        output_path_for_one_video = output_path
        labels_one_video = df_with_filenames[df_with_filenames["filename"]==video_filename.split(".")[0]]
        # extract frames from one video
        metadata = _extract_poses_from_one_video(path_to_video=path_to_video, detector=detector,
                                                 output_path=output_path_for_one_video, labels=labels_one_video,
                                  every_n_frame=every_n_frames)
        # concatenate obtained metadata for one video with the whole metadata (all videos)
        metadata_all = pd.concat([metadata_all, metadata], axis=0)
    return metadata_all


def main():
    # here we use the functions from the RECOLA.py file, since the preprocessing procedure is the same for SEMAINE as well
    # all we need is to modify paths to data, labels, and so on.
    path_to_data = "/media/external_hdd_2/Datasets/SEWA/Original/videos"
    output_path = "/media/external_hdd_2/Datasets/SEWA/preprocessed_pose"
    path_to_arousal_labels = "/media/external_hdd_2/Datasets/SEWA/SEW_labels_arousal_100Hz_gold_shifted.csv"
    path_to_valence_labels = "/media/external_hdd_2/Datasets/SEWA/SEW_labels_valence_100Hz_gold_shifted.csv"
    # load arousal and valence labels
    arousal_labels = load_labels(path_to_arousal_labels)
    arousal_labels.columns = ["filename", "timestamp", "arousal"]
    valence_labels = load_labels(path_to_valence_labels)
    valence_labels.columns = ["filename", "timestamp", "valence"]
    # merge arousal and valence labels
    labels = pd.merge(arousal_labels, valence_labels, on=["filename", "timestamp"])
    # change the dtype of timestamp column to float
    labels["timestamp"] = labels["timestamp"].astype(float)
    # extract poses
    metadata = extract_poses_with_filenames_in_dataframe(df_with_filenames=labels, path_to_data=path_to_data,
                                              output_path=output_path, every_n_frames=17)
    # delete nan values from the metadata
    metadata = metadata.dropna()
    # save metadata
    metadata.to_csv(os.path.join("/media/external_hdd_2/Datasets/SEWA/preprocessed_pose", "preprocessed_labels.csv"), index=False)



if __name__ == '__main__':
    main()