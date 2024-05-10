import sys

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

from emotion_recognition.datasets.facial.retinaface_utils import load_and_prepare_detector_retinaFace_mobileNet, \
    recognize_one_face_bbox, extract_face_according_bbox




def get_arousal_and_valence_for_every_frame_in_one_video(json_filename:str)->pd.DataFrame:
    result_df = pd.DataFrame(columns=['frame_num','arousal','valence'])
    with open(json_filename, "r") as f:
        data = json.load(f)
        data = data['frames']
        for key, value in data.items():
            result_df.loc[len(result_df.index)] = [key, value['arousal'], value['valence']]
    return result_df


def get_arousal_and_valence_all_videos(folder_path:str)->Dict[str, pd.DataFrame]:
    result_dict = {}
    for filename in os.listdir(folder_path):
        full_path_to_file = os.path.join(folder_path, filename, filename+ '.json')
        result_dict[full_path_to_file] = get_arousal_and_valence_for_every_frame_in_one_video(full_path_to_file)
        result_dict[full_path_to_file]['frame_num'] = result_dict[full_path_to_file]['frame_num'].\
            apply(lambda x: os.path.join(folder_path, filename, x+'.png'))
    return result_dict


def extract_faces_with_filenames_in_dataframe(dataframe_with_filenames:pd.DataFrame, output_path:str)->None:
    detector = load_and_prepare_detector_retinaFace_mobileNet()
    for idx in range(dataframe_with_filenames.shape[0]):
        filename = dataframe_with_filenames.iloc[idx, 0]
        new_filename = os.path.join(output_path, *filename.split(os.path.sep)[-2:])
        img = np.array(Image.open(filename))
        bbox = recognize_one_face_bbox(img, detector)
        if bbox is None:
            new_filename = np.NaN
        else:
            face = extract_face_according_bbox(img, bbox)
            os.makedirs(os.path.dirname(new_filename), exist_ok=True)
            Image.fromarray(face).save(new_filename)
        dataframe_with_filenames.iloc[idx, 0] = new_filename
    dataframe_with_filenames.to_csv(os.path.join(output_path, 'labels.csv'), index=False)



def main():
    # extract labels from json files
    path_dir = r'/media/external_hdd_1/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/data'
    labels = get_arousal_and_valence_all_videos(path_dir)
    # save labels for all videos as one dataframe
    labels_save_path = r'/media/external_hdd_1/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/preprocessed'
    if not os.path.exists(labels_save_path):
        os.makedirs(labels_save_path)
    labels = pd.concat(list(labels.values()), axis=0)
    labels.to_csv(os.path.join(labels_save_path, 'labels.csv'), index=False)

    # extract faces from videos
    output_path = r'/media/external_hdd_1/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/preprocessed'
    extract_faces_with_filenames_in_dataframe(labels, output_path)


if __name__ == "__main__":
    main()