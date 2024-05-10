import pandas as pd
import numpy as np
import sys
import os
import glob
sys.path.append("/work/home/dsu/engagement_recognition_project_server/")
sys.path.append("/work/home/dsu/datatools/")
from src.journalPaper.training.static.data_preparation import load_NoXi_and_DAiSEE_dataframes
from pytorch_utils.data_loaders.TemporalEmbeddingsLoader import TemporalEmbeddingsLoader
from pytorch_utils.data_loaders.TemporalLoadersStacker import TemporalLoadersStacker
from src.journalPaper.training.fusion.data_preparation import load_embeddings


paths_to_embeddings = glob.glob('/work/home/dsu/Datasets/Embeddings/*dev.csv')
paths_to_embeddings = {
    'DAiSEE_face_dev':paths_to_embeddings[0],
    'NoXi_face_dev':paths_to_embeddings[1],
    'DAiSEE_emo_dev':paths_to_embeddings[2],
    'DAiSEE_pose_dev':paths_to_embeddings[3],
    'NoXi_emo_dev':paths_to_embeddings[4],
    'NoXi_pose_dev':paths_to_embeddings[5],
        }
list_embeddings = load_embeddings(paths_to_embeddings)

data_loader = TemporalLoadersStacker(embeddings_with_labels_list=[list_embeddings['DAiSEE_face_dev'],
                                                            list_embeddings['DAiSEE_emo_dev'],
                                                            list_embeddings['DAiSEE_pose_dev']],
                                       label_columns=['label_0','label_1','label_2'],
                 window_size=4.0, stride=2.0,
                 consider_timestamps=True,
                 preprocessing_functions=None, shuffle=False)