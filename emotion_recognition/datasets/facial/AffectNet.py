import sys

import copy

import pandas as pd
import numpy as np
import os

from PIL import Image

from emotion_recognition.datasets.facial.retinaface_utils import load_and_prepare_detector_retinaFace_mobileNet, \
    recognize_one_face_bbox, extract_face_according_bbox


emo_categories:dict = {
    0:"N",
    1:"H",
    2:"Sa",
    3:"Su",
    4:"F",
    5:"D",
    6:"A",
    7:"C",
}

def change_labels_to_categories(df_with_labels:pd.DataFrame)->pd.DataFrame:
    result_df = df_with_labels
    result_df["category"] = result_df["category"].apply(lambda x: emo_categories[x])
    return result_df




def clear_df_with_subDirs_from_nonsense_labels(df_with_subDirs:pd.DataFrame)->pd.DataFrame:
    df_with_subDirs = df_with_subDirs[df_with_subDirs.iloc[:,-1] != 8]
    df_with_subDirs = df_with_subDirs[df_with_subDirs.iloc[:, -1] != 9]
    df_with_subDirs = df_with_subDirs[df_with_subDirs.iloc[:, -1] != 10]
    return df_with_subDirs


def clear_df_with_subDirs_from_NaN(df_with_subDirs:pd.DataFrame)->pd.DataFrame:
    df_with_subDirs = df_with_subDirs.dropna()
    return df_with_subDirs

def form_new_dataframe_using_original_one(original_df:pd.DataFrame, general_dir_with_data:str)->pd.DataFrame:
    # TODO: can be rewritten in vectorized way
    result_df_with_labels = pd.DataFrame(columns=['abs_path', 'arousal', 'valence', "category"])
    for idx in range(original_df.shape[0]):
        filename = original_df.iloc[idx, 0]
        abs_path_to_file = os.path.join(general_dir_with_data, *filename.split("/"))
        row_for_result_df = [abs_path_to_file, original_df.iloc[idx, -1], original_df.iloc[idx, -2], original_df.iloc[idx, -3]]
        result_df_with_labels.loc[len(result_df_with_labels.index)] = row_for_result_df
    return result_df_with_labels



def extract_faces_from_original_data(df_with_abs_paths:pd.DataFrame, output_path:str, labels_filename:str)->pd.DataFrame:
    result_df = copy.deepcopy(df_with_abs_paths)
    detector = load_and_prepare_detector_retinaFace_mobileNet()
    counter=0
    for idx in range(result_df.shape[0]):
        filename = result_df.iloc[idx, 0]
        # form absolute paths to images and faces to be extracted
        abs_output_path = os.path.join(output_path, *filename.split(os.path.sep)[-2:])
        try:
            os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
            # loading the image
            img = np.array(Image.open(filename))
            # recognize the face
            bbox = recognize_one_face_bbox(img, detector)
            # if not recognized, note it as NaN
            if bbox is None:
                abs_output_path = np.NaN
            else:
            # otherwise, extract the face and save it
                face = extract_face_according_bbox(img, bbox)
                Image.fromarray(face).save(abs_output_path)
            # change the filename to the created one
            result_df.iloc[idx, 0] = abs_output_path
        except Exception as e:
            print("During processing of file {} exception occured: {}".format(filename, e))

        if counter%100==0: print(f"Processed {counter} images")
        counter+=1

    result_df.to_csv(os.path.join(output_path, labels_filename), index=False)
    return result_df


def main():
    general_dir = "/media/external_hdd_1/Datasets/AffectNet/AffectNet/zip/Manually_Annotated_Images/"
    df_with_subDirs_train = pd.read_csv("/media/external_hdd_1/Datasets/AffectNet/AffectNet/zip/training.csv")
    df_with_subDirs_dev = pd.read_csv("/media/external_hdd_1/Datasets/AffectNet/AffectNet/zip/validation.csv")
    output_path = "/media/external_hdd_1/Datasets/AffectNet/AffectNet/preprocessed"
    # form absolute paths for every image in df
    df_with_subDirs_train = form_new_dataframe_using_original_one(df_with_subDirs_train, general_dir)
    df_with_subDirs_dev = form_new_dataframe_using_original_one(df_with_subDirs_dev, general_dir)
    # clear the dataframes from nonsense labels
    df_with_subDirs_train = clear_df_with_subDirs_from_nonsense_labels(df_with_subDirs_train)
    df_with_subDirs_dev = clear_df_with_subDirs_from_nonsense_labels(df_with_subDirs_dev)
    # extract faces from the original data
    df_with_subDirs_train = extract_faces_from_original_data(df_with_subDirs_train, output_path, "train_labels.csv")
    df_with_subDirs_dev = extract_faces_from_original_data(df_with_subDirs_dev, output_path, "dev_labels.csv")
    # clear the dataframes from NaNs
    df_with_subDirs_train = clear_df_with_subDirs_from_NaN(df_with_subDirs_train)
    df_with_subDirs_dev = clear_df_with_subDirs_from_NaN(df_with_subDirs_dev)
    # change the labels to categories
    df_with_subDirs_train = change_labels_to_categories(df_with_subDirs_train)
    df_with_subDirs_dev = change_labels_to_categories(df_with_subDirs_dev)
    # save the result
    df_with_subDirs_train.to_csv(os.path.join(output_path, "train_labels.csv"), index=False)
    df_with_subDirs_dev.to_csv(os.path.join(output_path, "dev_labels.csv"), index=False)

if __name__ == '__main__':
    main()


