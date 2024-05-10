import sys

import copy

import pandas as pd
import numpy as np
import os

from PIL import Image
from emotion_recognition.datasets.pose.hrnet_utils import cut_frame_to_pose
from simpleHRNet.SimpleHRNet import SimpleHRNet

emo_categories:dict = {
        1: "Su",
        2: "F",
        3: "D",
        4: "H",
        5: "Sa",
        6: "A",
        7: "N",
}



def change_labels_to_categories(df_with_labels:pd.DataFrame)->pd.DataFrame:
    result_df = df_with_labels
    result_df["category"] = result_df["category"].apply(lambda x: emo_categories[x])
    return result_df



def extract_faces_from_original_data(df_with_abs_paths:pd.DataFrame, output_path:str, labels_filename:str)->pd.DataFrame:
    result_df = copy.deepcopy(df_with_abs_paths)
    detector = SimpleHRNet(c=32, nof_joints=17, checkpoint_path="/work/home/dsu/simpleHigherHRNet/pose_higher_hrnet_w32_512.pth",
                                 return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1, device="cuda")
    counter=0
    for idx in range(result_df.shape[0]):
        filename = result_df.iloc[idx, 0]
        # form absolute paths to images and faces to be extracted
        abs_output_path = os.path.join(output_path, *filename.split(os.path.sep)[-1:])
        try:
            os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
            # loading the image
            img = np.array(Image.open(filename))
            # recognize the face
            pose = cut_frame_to_pose(extractor=detector, frame=img, return_bbox=False)
            # if not recognized, note it as NaN
            if pose is None:
                abs_output_path = np.NaN
            else:
            # otherwise, extract the face and save it
                Image.fromarray(pose).save(abs_output_path)
            # change the filename to the created one
            result_df.iloc[idx, 0] = abs_output_path
        except Exception as e:
            print("During processing of file {} exception occured: {}".format(filename, e))

        if counter%1000==0: print(f"Processed {counter} images")
        counter+=1

    result_df.to_csv(os.path.join(output_path, labels_filename), index=False)
    return result_df

def form_new_dataframe_using_original_one(original_df:pd.DataFrame, general_dir_with_data:str)->pd.DataFrame:
    # TODO: rewrite in vectorized way
    result_df_with_labels = pd.DataFrame(columns=['abs_path', 'arousal', 'valence', "category"])
    for idx in range(original_df.shape[0]):
        filename = original_df.iloc[idx, 0]
        abs_path_to_file = os.path.join(general_dir_with_data, *filename.split("/"))
        row_for_result_df = [abs_path_to_file, original_df.iloc[idx, -3], original_df.iloc[idx, -2], original_df.iloc[idx, -1]]
        result_df_with_labels.loc[len(result_df_with_labels.index)] = row_for_result_df
    return result_df_with_labels



def main():
    path_to_data = "/media/external_hdd_2/Datasets/RAF_DB/Image/original/"
    path_to_labels = "/media/external_hdd_2/Datasets/RAF_DB/EmoLabel/list_patition_label.txt"
    path_to_output = "/media/external_hdd_2/Datasets/RAF_DB/preprocessed_pose/"
    # load labels and prepare them, transforming to the common format
    labels = pd.read_csv(path_to_labels, sep=" ", header=None)
    labels.columns = ['filename', 'category']
    labels = change_labels_to_categories(labels)
    labels["arousal"] = np.NaN
    labels["valence"] = np.NaN
    # reorder columns
    labels = labels[['filename', 'arousal', 'valence', 'category']]
    # form new dataframe with absolute paths to images
    new_df_with_labels = form_new_dataframe_using_original_one(labels, path_to_data)
    # extract faces from images and save them
    new_df_with_labels = extract_faces_from_original_data(new_df_with_labels, path_to_output, "labels.csv")
    # drop lines with NaNs in the "abs_path" column
    new_df_with_labels = new_df_with_labels.dropna(subset=["abs_path"])
    # split the data into train and test
    train_part = new_df_with_labels[new_df_with_labels["abs_path"].str.contains("train")]
    test_part = new_df_with_labels[new_df_with_labels["abs_path"].str.contains("test")]
    train_part.to_csv(os.path.join(path_to_output, "train_labels.csv"), index=False)
    test_part.to_csv(os.path.join(path_to_output, "test_labels.csv"), index=False)





if __name__ == "__main__":
    main()