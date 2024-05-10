import sys
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

from emotion_recognition.datasets.pose.hrnet_utils import cut_frame_to_pose
from simpleHRNet.SimpleHRNet import SimpleHRNet


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
    detector = SimpleHRNet(c=32, nof_joints=17, checkpoint_path="/work/home/dsu/simpleHigherHRNet/pose_hrnet_w32_256x192.pth",
                                 return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1, device="cuda")
    for idx in range(dataframe_with_filenames.shape[0]):
        filename = dataframe_with_filenames.iloc[idx, 0]
        new_filename = os.path.join(output_path, *filename.split(os.path.sep)[-2:])
        img = np.array(Image.open(filename))
        pose = cut_frame_to_pose(extractor=detector, frame=img, return_bbox=False)
        if pose is None:
            new_filename = np.NaN
        else:
            os.makedirs(os.path.dirname(new_filename), exist_ok=True)
            Image.fromarray(pose).save(new_filename)
        dataframe_with_filenames.iloc[idx, 0] = new_filename
        if idx % 1000 == 0:
            print(f"Processed {idx} images, {dataframe_with_filenames.shape[0] - idx} left...")
    dataframe_with_filenames.to_csv(os.path.join(output_path, 'labels.csv'), index=False)



def main():
    # extract labels from json files
    path_dir = r'/media/external_hdd_2/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/data'
    labels = get_arousal_and_valence_all_videos(path_dir)
    # save labels for all videos as one dataframe
    labels_save_path = r'/media/external_hdd_2/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/preprocessed_pose'
    if not os.path.exists(labels_save_path):
        os.makedirs(labels_save_path)
    labels = pd.concat(list(labels.values()), axis=0)
    labels.to_csv(os.path.join(labels_save_path, 'labels.csv'), index=False)

    # extract faces from videos
    output_path = r'/media/external_hdd_2/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/preprocessed_pose'
    extract_faces_with_filenames_in_dataframe(labels, output_path)


if __name__ == "__main__":
    main()