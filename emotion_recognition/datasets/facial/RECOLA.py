import sys

import glob
import os
from typing import Optional

import pandas as pd
import numpy as np
from PIL import Image
import cv2

from emotion_recognition.datasets.facial.retinaface_utils import load_and_prepare_detector_retinaFace_mobileNet, \
    recognize_one_face_bbox, extract_face_according_bbox


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



def _extract_faces_from_one_video(path_to_video:str, detector:object, output_path:str, labels:pd.DataFrame,
                                  every_n_frame:Optional[int]=1)->pd.DataFrame:
    """ Extracts faces from one video and saves them to the output path. Also, generates the metadata for the extracted
        faces, which is returned by this function.

    :param path_to_video: str
        path to the video file
    :param detector: object
        Face detector, RetinafaceDetector
    :param output_path: str
        path to the output folder, where the extracted faces will be saved
    :param labels: pd.DataFrame
        dataframe with labels with the following columns: ["filename", "timestamp", "arousal", "valence"]
    :param every_n_frame: int
        extract faces from every n-th frame
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
                # recognize the face
                bbox = recognize_one_face_bbox(frame, detector)
                # if not recognized, note it as NaN
                if bbox is None:
                    output_filename = np.NaN
                else:
                    # otherwise, extract the face and save it
                    face = extract_face_according_bbox(frame, bbox)
                    output_filename = os.path.join(output_path, path_to_video.split(os.path.sep)[-1].split(".")[0]
                                                   + f"_%s.png"%(str(timestamp).replace(".", "_")))
                    # save extracted face
                    Image.fromarray(face).save(output_filename)
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





def extract_faces_with_filenames_in_dataframe(df_with_filenames:pd.DataFrame, path_to_data:str, output_path:str,
                                              every_n_frames:Optional[int]=1)->pd.DataFrame:
    """ Extracts faces from the videos in the dataframe (and available through the path_to_data) and saves them to
    the output path. Also, generates the metadata for all extracted faces, which is being returned by this function.

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
    detector = load_and_prepare_detector_retinaFace_mobileNet()
    # search for unique video filenames in provided dataframe
    unique_video_filenames = df_with_filenames["filename"].unique()
    # go through different videos
    for video_filename in unique_video_filenames:
        # create variables needed to run _extract_faces_from_one_video() function
        video_filename = os.path.basename(glob.glob(os.path.join(path_to_data, video_filename + "*"))[0])
        path_to_video = os.path.join(path_to_data, video_filename)
        output_path_for_one_video = output_path
        labels_one_video = df_with_filenames[df_with_filenames["filename"]==video_filename.split(".")[0]]
        # extract frames from one video
        metadata = _extract_faces_from_one_video(path_to_video=path_to_video, detector=detector,
                                                 output_path=output_path_for_one_video, labels=labels_one_video,
                                  every_n_frame=every_n_frames)
        # concatenate obtained metadata for one video with the whole metadata (all videos)
        metadata_all = pd.concat([metadata_all, metadata], axis=0)
    return metadata_all



def main():
    path_to_data = "/media/external_hdd_2/Datasets/RECOLA/original/RECOLA_Video_recordings/"
    output_path = "/media/external_hdd_2/Datasets/RECOLA/preprocessed"
    path_to_arousal_labels = "/media/external_hdd_2/Datasets/RECOLA/REC_labels_arousal_100Hz_gold_shifted.csv"
    path_to_valence_labels = "/media/external_hdd_2/Datasets/RECOLA/REC_labels_valence_100Hz_gold_shifted.csv"
    # load arousal and valence labels
    arousal_labels = load_labels(path_to_arousal_labels)
    arousal_labels.columns = ["filename", "timestamp", "arousal"]
    valence_labels = load_labels(path_to_valence_labels)
    valence_labels.columns = ["filename", "timestamp", "valence"]
    # merge arousal and valence labels
    labels = pd.merge(arousal_labels, valence_labels, on=["filename", "timestamp"])
    # change the filenames of the videofile to the original ones (REC00NN -> PNN)
    labels["filename"] = labels["filename"].apply(lambda x: x.replace("REC00", "P"))
    # change the dtype of timestamp column to float
    labels["timestamp"] = labels["timestamp"].astype(float)
    # extract faces
    metadata = extract_faces_with_filenames_in_dataframe(df_with_filenames=labels, path_to_data=path_to_data,
                                              output_path=output_path, every_n_frames=8)
    # delete nan values from the metadata
    metadata = metadata.dropna()
    # save metadata
    metadata.to_csv(os.path.join("/media/external_hdd_2/Datasets/RECOLA/preprocessed", "preprocessed_labels.csv"), index=False)



if __name__ == '__main__':
    main()