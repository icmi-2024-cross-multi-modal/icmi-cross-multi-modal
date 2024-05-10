import sys
from typing import Optional



import copy

import pandas as pd
import numpy as np
import os

from PIL import Image

from emotion_recognition.datasets.facial.retinaface_utils import load_and_prepare_detector_retinaFace_mobileNet, \
    recognize_one_face_bbox, extract_face_according_bbox

emo_categories:dict = {
    0:"A",
    1:"D",
    2:"F",
    3:"H",
    4:"Sa",
    5:"Su",
    6:"N",
}

def change_labels_to_categories(df_with_labels:pd.DataFrame)->pd.DataFrame:
    result_df = df_with_labels
    result_df["category"] = result_df["category"].apply(lambda x: emo_categories[x])
    return result_df


def tranform_dataframe_with_pixel_values_to_images(df_with_pixel_values:pd.DataFrame, save_images:bool=True,
                                                    path_to_output:Optional[str]=None)->pd.DataFrame:
    # create a folder if it does not exist
    if save_images:
        if not os.path.exists(path_to_output):
            os.makedirs(path_to_output)
    abs_paths = pd.DataFrame(columns=["abs_path"])
    # iterate over all rows in the dataframe
    for idx in range(df_with_pixel_values.shape[0]):
        # get pixel values. THey are stored as string, where each pixel is separated by space
        pixels = df_with_pixel_values.iloc[idx,0]
        # split the string to get a list of pixel values
        pixels = pixels.split(" ")
        # convert the list of strings to the list of integers
        pixels = [int(x) for x in pixels]
        # convert the list of integers to the numpy array
        pixels = np.array(pixels)
        # reshape the array to the 48x48 image
        pixels = pixels.reshape((48,48))
        # convert it to uint8
        pixels = pixels.astype(np.uint8)
        # convert the array to the image
        img = Image.fromarray(pixels)
        # save the image
        if save_images:
            filename = "img_{}.jpg".format(idx)
            img.save(os.path.join(path_to_output, filename))
            # save the absolute path to the image
            abs_paths.loc[idx] = os.path.join(path_to_output, filename)

    return abs_paths


def extract_faces_from_original_data(df_with_abs_paths:pd.DataFrame, output_path:str, labels_filename:str)->pd.DataFrame:
    result_df = copy.deepcopy(df_with_abs_paths)
    detector = load_and_prepare_detector_retinaFace_mobileNet()
    counter=0
    for idx in range(result_df.shape[0]):
        filename = result_df.iloc[idx, 0]
        # form absolute paths to images and faces to be extracted
        abs_output_path = os.path.join(output_path, *filename.split(os.path.sep)[-1:])
        try:
            os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
            # loading the image
            img = np.array(Image.open(filename).convert("RGB"))
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
    data_and_labels_path = "/media/external_hdd_2/Datasets/FER_plus/train.csv"
    path_to_original_data = "/media/external_hdd_2/Datasets/FER_plus/images/"
    path_to_output = "/media/external_hdd_2/Datasets/FER_plus/preprocessed/"
    # load data and labels. THey are represented in the FER+ dataset as a single csv file, where
    # each row is a label with a corresponding pixel values of that image
    df_with_labels = pd.read_csv(data_and_labels_path, header=0, sep=",")
    # separate labels from data and transform them to the common format
    labels = df_with_labels.drop(columns=["pixels"])
    labels.columns = ["category"]
    labels = change_labels_to_categories(labels)
    # save pixel values as images
    df_with_labels = df_with_labels.drop(columns=["emotion"])
    df_with_labels = tranform_dataframe_with_pixel_values_to_images(df_with_labels, save_images=True,
                                                                    path_to_output=path_to_original_data)
    # combine labels and paths to images
    df_with_labels = pd.concat([df_with_labels, labels], axis=1)
    # complete the df_with_labels with arousal and valence
    df_with_labels["arousal"] = np.NaN
    df_with_labels["valence"] = np.NaN
    df_with_labels = df_with_labels[['abs_path', 'arousal', 'valence', 'category']]
    # save final labels with abs paths
    print("Dropped {} images".format(df_with_labels["abs_path"].isna().sum()))
    df_with_labels = df_with_labels.dropna(subset=["abs_path"])
    df_with_labels.to_csv(os.path.join(path_to_original_data, "labels.csv"), index=False)




if __name__ == "__main__":
    main()