import sys


import copy

import pandas as pd
import numpy as np
import os

from PIL import Image

from emotion_recognition.datasets.facial.retinaface_utils import load_and_prepare_detector_retinaFace_mobileNet, \
    recognize_one_face_bbox, extract_face_according_bbox

emo_categories:dict = {
    0: "A",
    1: "D",
    2: "F",
    3: "H",
    4: "Sa",
    5: "Su",
    6: "N",
}


def change_labels_to_categories(df_with_labels:pd.DataFrame)->pd.DataFrame:
    result_df = df_with_labels
    result_df["category"] = result_df["category"].apply(lambda x: emo_categories[x])
    return result_df

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
    original_labels_columns = ["image_name", "face_id_in_image", "face_box_top", "face_box_left", "face_box_right",
                               "face_box_bottom", "face_box_cofidence", "expression_label"]
    path_to_data = "/media/external_hdd_2/Datasets/ExpW/ExpW/data/image/origin/"
    path_to_labels = "/media/external_hdd_2/Datasets/ExpW/ExpW/label/label.lst"
    path_to_output = "/media/external_hdd_2/Datasets/ExpW/ExpW/preprocessed/"
    # load labels
    df_with_labels = pd.read_csv(path_to_labels, sep=" ", header=None, names=original_labels_columns)
    # delete rows with more than one face in the image
    print("The number of rows with more than one face in the image (will be dropped): {}".format(
        df_with_labels[df_with_labels["face_id_in_image"] != 0].shape[0]))
    df_with_labels = df_with_labels[df_with_labels["face_id_in_image"] == 0]
    # prepare labels for further processing
    df_with_labels = df_with_labels.drop(columns=["face_id_in_image", "face_box_top", "face_box_left", "face_box_right",
                                                    "face_box_bottom", "face_box_cofidence"])
    df_with_labels.columns = ["abs_path", "category"]
    # change labels to str categories
    df_with_labels = change_labels_to_categories(df_with_labels)
    # form absolute paths to images
    df_with_labels["abs_path"] = df_with_labels["abs_path"].apply(lambda x: os.path.join(path_to_data, x))

    # extract faces from original data
    df_with_labels = extract_faces_from_original_data(df_with_labels, path_to_output, "labels.csv")

    # transform labels to the common format and save it
    df_with_labels["valence"] = np.NaN
    df_with_labels["arousal"] = np.NaN
    df_with_labels = df_with_labels[['abs_path', 'arousal', 'valence', 'category']]
    # delete rowns wiw NaNs in abs_path
    print("The number of rows with NaNs in abs_path (will be dropped): {}".format(df_with_labels["abs_path"].isna().sum()))
    df_with_labels = df_with_labels.dropna(subset=["abs_path"])

    df_with_labels.to_csv(os.path.join(path_to_output, "labels.csv"), index=False)










if __name__ == "__main__":
    main()