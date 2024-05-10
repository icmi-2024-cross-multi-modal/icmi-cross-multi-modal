import sys
from typing import Tuple


import scipy.io
import copy
import re
import pandas as pd
import numpy as np
import os

from PIL import Image

from emotion_recognition.datasets.facial.retinaface_utils import load_and_prepare_detector_retinaFace_mobileNet, \
    recognize_one_face_bbox, extract_face_according_bbox




def extract_labels_from_mat_file(mat_file_path:str)->Tuple[pd.DataFrame,...]:
    mat_file = scipy.io.loadmat(mat_file_path)
    train = mat_file['train']
    test = mat_file['test']
    val = mat_file['val']
    train = extract_labels_from_struct_array(train)
    test = extract_labels_from_struct_array(test)
    val = extract_labels_from_struct_array(val)

    return train, val, test





def extract_labels_from_struct_array(array:object)->pd.DataFrame:
    final_labels = pd.DataFrame(columns=['filename', 'arousal', 'valence', 'category_0', 'category_1', 'category_2'])
    drop_counter = 0
    emo_categories_pattern = '\[\'\D*\'\]'
    emo_categories_pattern = re.compile(emo_categories_pattern)
    arousal_valence_pattern = '\[\d*\]'
    arousal_valence_pattern = re.compile(arousal_valence_pattern)
    for item in array[0]:
        filename = str(item['filename'][0])
        categories = item['person']['annotations_categories'].flatten()
        if len(categories)!=1:
            drop_counter+=1
            continue
        # preprocess categories from string to list of values
        categories = str(categories[0].flatten())
        categories = emo_categories_pattern.findall(categories)
        categories = [category[2:-2] for category in categories]  # remove [' and '] from the beginning and the end
        # check if there are less than 3 categories
        if len(categories) < 3:
            # add nan values to the list
            categories.extend([np.nan] * (3 - len(categories)))
        # search for the arousal and valence values
        arousal_valence = item['person']['annotations_continuous'].flatten()
        arousal_valence = str(arousal_valence.flatten()) # the order of values is : valence, arousal, dominance
        arousal_valence = arousal_valence_pattern.findall(arousal_valence)
        arousal_valence = [category[1:-1] for category in arousal_valence]  # remove [ and ] from the beginning and the end
        valence = float(arousal_valence[0])
        arousal = float(arousal_valence[1])
        # create a new row in the dataframe
        new_row = {'filename': filename, 'arousal': arousal, 'valence': valence, 'category_0': categories[0],
                      'category_1': categories[1], 'category_2': categories[2]}
        # append row to the dataframe
        final_labels.loc[len(final_labels)] = new_row
    print(f'Dropped {drop_counter} rows')
    return final_labels


def form_new_dataframe_using_original_one(original_df:pd.DataFrame, general_dir_with_data:str)->pd.DataFrame:
    # TODO: can be rewritten in vectorized way
    result_df_with_labels = pd.DataFrame(columns=['abs_path', 'arousal', 'valence', "category"])
    for idx in range(original_df.shape[0]):
        filename = original_df.iloc[idx, 0]
        abs_path_to_file = os.path.join(general_dir_with_data, *filename.split("/"))
        row_for_result_df = [abs_path_to_file, original_df['arousal'].iloc[idx], original_df['valence'].iloc[idx], original_df['category'].iloc[idx]]
        result_df_with_labels.loc[len(result_df_with_labels.index)] = row_for_result_df
    return result_df_with_labels


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
            # transform graz image to RGB if needed
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)
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





def get_one_category_from_three(dataframe_with_categories:pd.DataFrame)->pd.DataFrame:
    classic_categories = ['Happiness', 'Surprise', 'Aversion', 'Anger', 'Sadness', 'Fear']
    category_translation = {
        'Happiness': 'H',
        'Surprise': 'Su',
        'Aversion': 'D',
        'Anger': 'A',
        'Sadness': 'Sa',
        'Fear': 'F',
        # these are other catgories from EMOTIC, which will be injected into the classical 6 categories
        'Annoyance': 'A',
        'Peace': 'N',
        'Affection': 'H',
        'Pleasure': 'H',
        'Excitement': 'H',
        'Disconnection': 'Sa',
        'Yearning': 'A',
        'Disapproval': 'C',
        # categories, which should be dropped out
        'Esteem': None,
        'Anticipation': None,
        'Engagement': None,
        'Confidence': None,
        'Sympathy': None,
        'Doubt/Confusion': None,
        'Fatigue': None,
        'Embarrassment': None,
        'Sensitivity': None,
        'Disquietment': None,
        'Pain': None,
        'Suffering': None,
    }
    # create a new columns in the dataframe with final category
    dataframe_with_categories['category'] = np.nan
    # iterate over all rows in the dataframe
    for idx in range(dataframe_with_categories.shape[0]):
        # get list of categories for the current row
        categories = dataframe_with_categories.iloc[idx, 3:6].tolist()
        # get rid of nan values
        categories = [category for category in np.array(categories) if category != 'nan']
        # check if there is at least one category from the classical 6 categories
        if any([category in classic_categories for category in categories]):
            # if yes, we take the first possible category encountered in categories list
            for category in categories:
                if category in classic_categories:
                    dataframe_with_categories.iloc[idx, -1] = category_translation[category]
                    break
        else:
            # if not, we check if other not NaN categories are present
            if any([category_translation[category] is not None for category in categories]):
                # if yes, we take the first possible category encountered in categories list
                for category in categories:
                    if category_translation[category] is not None:
                        dataframe_with_categories.iloc[idx, -1] = category_translation[category]
                        break
            else:
                # if not, just assign NaN
                dataframe_with_categories.iloc[idx, -1] = np.NaN

    return dataframe_with_categories



def main():
    path_to_labels= '/media/external_hdd_2/Datasets/EMOTIC/CVPR17_Annotations.mat'
    path_to_data = '/media/external_hdd_2/Datasets/EMOTIC/cvpr_emotic/cvpr_emotic/images/'
    output_path = '/media/external_hdd_2/Datasets/EMOTIC/cvpr_emotic/cvpr_emotic/preprocessed/'
    # extract labels from .mat file
    train_labels, val_labels, test_labels = extract_labels_from_mat_file(path_to_labels)

    # transform labels to common format
    train_labels = get_one_category_from_three(train_labels)
    val_labels = get_one_category_from_three(val_labels)
    test_labels = get_one_category_from_three(test_labels)

        # drop columns with three categories
    train_labels.drop(columns=['category_0', 'category_1', 'category_2'], inplace=True)
    val_labels.drop(columns=['category_0', 'category_1', 'category_2'], inplace=True)
    test_labels.drop(columns=['category_0', 'category_1', 'category_2'], inplace=True)

        # drop NaN values placed in the category column
    print('train: {} rows have been dropped'.format(train_labels.shape[0] - train_labels.dropna(subset=['category']).shape[0]))
    print('val: {} rows have been dropped'.format(val_labels.shape[0] - val_labels.dropna(subset=['category']).shape[0]))
    print('test: {} rows have been dropped'.format(test_labels.shape[0] - test_labels.dropna(subset=['category']).shape[0]))
    train_labels = train_labels.dropna(subset=['category'])
    val_labels = val_labels.dropna(subset=['category'])
    test_labels = test_labels.dropna(subset=['category'])

        # normalize valence and arousal. They are in range [1, 10], while should be in range [-1, 1]
    train_labels['valence'] = (train_labels['valence'] - 5.5) / 4.5
    train_labels['arousal'] = (train_labels['arousal'] - 5.5) / 4.5
    val_labels['valence'] = (val_labels['valence'] - 5.5) / 4.5
    val_labels['arousal'] = (val_labels['arousal'] - 5.5) / 4.5
    test_labels['valence'] = (test_labels['valence'] - 5.5) / 4.5
    test_labels['arousal'] = (test_labels['arousal'] - 5.5) / 4.5

        # form new dataframe by adding absolute path to the image
    train_labels = form_new_dataframe_using_original_one(train_labels, path_to_data)
    val_labels = form_new_dataframe_using_original_one(val_labels, path_to_data)
    test_labels = form_new_dataframe_using_original_one(test_labels, path_to_data)

    # extract faces from images
    train_labels = extract_faces_from_original_data(train_labels, output_path, 'train_labels.csv')
    val_labels = extract_faces_from_original_data(val_labels, output_path, 'val_labels.csv')
    test_labels = extract_faces_from_original_data(test_labels, output_path, 'test_labels.csv')

    # drop nan values from the dataframe
    print('After face extraction train: {} rows have been dropped'.format(train_labels.shape[0] - train_labels.dropna(subset=['abs_path']).shape[0]))
    print('After face extraction val: {} rows have been dropped'.format(val_labels.shape[0] - val_labels.dropna(subset=['abs_path']).shape[0]))
    print('After face extraction test: {} rows have been dropped'.format(test_labels.shape[0] - test_labels.dropna(subset=['abs_path']).shape[0]))
    train_labels = train_labels.dropna(subset=['abs_path'])
    val_labels = val_labels.dropna(subset=['abs_path'])
    test_labels = test_labels.dropna(subset=['abs_path'])

    # save dataframes to csv files
    train_labels.to_csv(os.path.join(output_path, 'train_labels.csv'), index=False)
    val_labels.to_csv(os.path.join(output_path, 'val_labels.csv'), index=False)
    test_labels.to_csv(os.path.join(output_path, 'test_labels.csv'), index=False)








if __name__ == "__main__":
    main()