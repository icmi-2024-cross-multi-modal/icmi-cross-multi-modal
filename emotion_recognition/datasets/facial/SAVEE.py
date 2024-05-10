import glob
import sys

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from emotion_recognition.datasets.facial.retinaface_utils import load_and_prepare_detector_retinaFace_mobileNet, \
    recognize_one_face_bbox, extract_face_according_bbox


import os

import pandas as pd
import re


def extract_faces_with_filenames_in_dataframe(df_with_filenames:pd.DataFrame, output_path:str,
                                              every_n_frames:Optional[int]=1)->pd.DataFrame:
    """ Extracts faces from the videos in the dataframe (and available through the path_to_data) and saves them to
    the output path. Also, generates the metadata for all extracted faces, which is being returned by this function.

    :param df_with_filenames:
    :param output_path:
    :param every_n_frames:
    :return:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create the metadata dataframe to store the paths to every frame and corresponding labels
    metadata_all=pd.DataFrame(columns=["filename", "timestamp", "arousal", "valence", "category"])
    detector = load_and_prepare_detector_retinaFace_mobileNet()
    # go through different videos
    for path_to_video in df_with_filenames['abs_path'].unique():
        # create variables needed to run _extract_faces_from_one_video() function
        video_filename = os.path.basename(path_to_video)
        output_adding = path_to_video.split(os.path.sep)[-2:]
        output_adding[-1] = output_adding[-1].split(".")[0]
        output_path_for_one_video = os.path.join(output_path, *output_adding)
        if not os.path.exists(output_path_for_one_video):
            os.makedirs(output_path_for_one_video, exist_ok=True)
        labels_one_video = df_with_filenames[df_with_filenames["abs_path"]==path_to_video]
        # extract frames from one video
        metadata = _extract_faces_from_one_video(path_to_video=path_to_video, detector=detector,
                                                 output_path=output_path_for_one_video, labels=labels_one_video,
                                  every_n_frame=every_n_frames)
        metadata["category"] = labels_one_video["category"].values[0]
        # concatenate obtained metadata for one video with the whole metadata (all videos)
        metadata_all = pd.concat([metadata_all, metadata], axis=0)
    return metadata_all

def _extract_faces_from_one_video(path_to_video:str, detector:object, output_path:str, labels:pd.DataFrame,
                                  every_n_frame:Optional[int]=1)->pd.DataFrame:
    # metadata
    metadata = pd.DataFrame(columns=["filename", "timestamp", "arousal", "valence", "category"])
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
                metadata = pd.concat([metadata,
                                      pd.DataFrame.from_records([{
                                            "filename": output_filename,
                                            "timestamp": timestamp,
                                            "arousal": np.NaN,
                                            "valence": np.NaN,
                                            "category": np.NaN
                                      }])
                                      ], ignore_index=True)
            # increment counter
            counter += 1
        else:
            break
    return metadata



def generate_paths_and_labels_from_dir(path_to_dir:str)->pd.DataFrame:
    emo_categories = {
        'a':'A',
        'd':'D',
        'f':'F',
        'h':'H',
        'n':'N',
        'sa':'Sa',
        'su':'Su',
    }
    # create dataframe with paths to data and labels
    columns = ["abs_path", 'arousal', 'valence', 'category']
    df = pd.DataFrame(columns=columns)
    # get all paths to files based on the glob.glob patern
    paths_to_files = glob.glob(os.path.join(path_to_dir,'*', "*.avi"))
    # define the pattern to extract label from filename
    pattern = re.compile(r'^\D*')
    # iterate over all paths to files
    for path_to_file in paths_to_files:
        # get filename
        filename = os.path.basename(path_to_file)
        # get label from filename
        label = pattern.match(filename).group()
        # get arousal and valence from label
        arousal = np.NaN
        valence = np.NaN
        # get category from label
        category = emo_categories[label]
        # append row to dataframe
        df.loc[len(df.index)] = [path_to_file, arousal, valence, category]

    return df



def main():
    # params
    path_to_data = "/media/external_hdd_2/Datasets/SAVEE/AudioVisualClip/AudioVisualClip/"
    output_path = "/media/external_hdd_2/Datasets/SAVEE/preprocessed/"
    every_n_frames = 20
    # generate paths to videos and labels just using the directory structure
    paths_with_labels = generate_paths_and_labels_from_dir(path_to_data)
    # extract faces from videos and save them to the output path
    paths_with_labels = extract_faces_with_filenames_in_dataframe(df_with_filenames=paths_with_labels, output_path=output_path,
    every_n_frames = every_n_frames)
    # rename columns
    paths_with_labels = paths_with_labels.rename(columns={"filename": "abs_path"})

    # save the dataframe with paths to faces and labels
    paths_with_labels.to_csv(os.path.join(output_path, "labels.csv"), index=False)





if __name__ == '__main__':
    main()