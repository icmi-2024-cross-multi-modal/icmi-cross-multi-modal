from typing import Dict, List, Callable, Tuple, Union

import torch
from torch import nn

from Models.model_classes import get_engagement_model_efficientNet_b1, get_engagement_model_HRNet, \
    uni_modal_dynamic_facial_model, uni_modal_dynamic_kinesics_model, get_preprocessing_functions, \
    get_emo_model_efficientNet_b1, AttentionFusionModel_2dim, AttentionFusionModel_3dim_v1





class Uni_modal_ER_model(nn.Module):
    def __init__(self, config:Dict[str, str]):
        super(Uni_modal_ER_model, self).__init__()
        self.model_config = config
        if "Static" in self.model_config["model_type"]:
            self.static_model = self.__initialize_static_model(self.model_config["model_type"],
                                                               self.model_config["static_model_path"][0])
            self.dynamic_model = None
        elif "Dynamic" in self.model_config["model_type"]:
            self.static_model = self.__initialize_static_model(self.model_config["model_type"],
                                                                self.model_config["static_model_path"][0])
            # cut off the last layer of the static model
            self.static_model = nn.Sequential(*list(self.static_model.children())[:-1])
            self.dynamic_model = self.__initialize_dynamic_model(self.model_config["model_type"],
                                                                 self.model_config["dynamic_model_path"],
                                                                 self.static_model)
        # define preprocessing functions
        self.preprocessing_functions = get_preprocessing_functions(model_type= "Modified_HRNet" if "kinesics" in self.model_config["model_type"]
                                                                                else "EfficientNet-B1")

    def __initialize_static_model(self, model_type:str, path_to_weights):
        if "facial" in model_type:
            model = get_engagement_model_efficientNet_b1(path_to_weights=path_to_weights)
        elif "kinesics" in model_type:
            model = get_engagement_model_HRNet(path_to_weights=path_to_weights,
                                               path_to_HRNet_weights=self.model_config["hrnet_weights_path"])
        else:
            raise ValueError("Invalid model type. Only 'Static_facial_ER' and 'Static_kinesics_ER' are supported for "
                             "the static model type.")
        model.load_state_dict(torch.load(path_to_weights))
        model.eval()
        return model

    def __initialize_uni_modal_dynamic_model(self, model_type:str, path_to_weights, static_model:nn.Module):
        if model_type == "Dynamic_uni_modal_facial_ER":
            model = uni_modal_dynamic_facial_model(facial_model=static_model, embeddings_layer_neurons=256,
                 num_classes=3, transformer_num_heads=4, num_timesteps=8*5) # 8 seconds with 5 FPS
        elif model_type == "Dynamic_uni_modal_kinesics_ER":
            model = uni_modal_dynamic_kinesics_model(pose_model=static_model, embeddings_layer_neurons=256,
                                                    num_classes=3, transformer_num_heads=4, num_timesteps=8*5) # 8 seconds with 5 FPS
        else:
            raise ValueError("Invalid model type. Only 'Dynamic_uni_modal_facial_ER' and 'Dynamic_uni_modal_kinesics_ER' "
                             "are supported for the dynamic model type.")
        model.load_state_dict(torch.load(path_to_weights))
        model.eval()
        return model


    def __preprocess(self, x:torch.Tensor)->torch.Tensor:
        # x can have shape (batch_size, channels, height, width) or (batch_size, num_timesteps, channels, height, width)
        # if the model is static, we need to preprocess the whole batch
        if "Static" in self.model_config["model_type"]:
            x = torch.stack([self.preprocessing_functions(image) for image in x])
        # if the model is dynamic, we need to preprocess every timestep
        else:
            x = torch.stack([torch.stack([self.preprocessing_functions(image) for image in timestep]) for timestep in x])
        return x

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # preprocess the input
        x = self.__preprocess(x)
        # pass the input through the model
        if "Static" in self.model_config["model_type"]:
            return self.static_model(x)
        elif "Dynamic" in self.model_config["model_type"]:
            return self.dynamic_model(x)
        return x

class Multi_modal_ER_model(nn.Module):

    def __init__(self, config:Dict[str, Union[str, List]]):
        super(Multi_modal_ER_model, self).__init__()
        self.model_config = config
        # parse the type of the static extractors from model_type
        self.static_model_types = self.model_config["model_type"].split("_")
        self.static_model_types = [model_type for model_type in self.static_model_types
                                   if model_type in ["facial", "kinesics", "affective", "all"]]
        # !!! important: order of the modalities is important. Here is the listed order for different combinations:
        #   2-modal: # affective - facial        # affective - kinesics     # kinesics - facial
        #   3-modal: # facial - kinesics - affective
        # create static extractors
        self.static_models = self.__initialize_static_models(self.static_model_types, self.model_config["static_model_paths"])
        # create dynamic model
        self.dynamic_model = self.__initialize_dynamic_model(self.model_config["model_type"], self.model_config["dynamic_model_path"])
        # define preprocessing functions
        self.list_preprocessing_functions = []
        for model_type in self.static_model_types:
            prep_model_type = "Modified_HRNet" if "kinesics" in model_type else "EfficientNet-B1"
            self.list_preprocessing_functions.append(get_preprocessing_functions(model_type=prep_model_type))



    def __initialize_static_models(self, model_types:List[str], paths_to_weights:List[str])->List[nn.Module]:
        static_models = []
        for model_type, path in zip(model_types, paths_to_weights):
            if model_type == "facial":
                model = get_engagement_model_efficientNet_b1(path_to_weights=path)
            elif model_type == "kinesics":
                model = get_engagement_model_HRNet(path_to_weights=path,
                                                   path_to_HRNet_weights=self.model_config["hrnet_weights_path"])
            elif model_type == "affective":
                model = get_emo_model_efficientNet_b1(path_to_weights=path)
            else:
                raise ValueError("Invalid model type. Only 'facial' and 'kinesics' are supported for the static model type.")
            model.load_state_dict(torch.load(path))
            # cut off last or last two layers (if model is affective) to get a feature extractor
            model = nn.Sequential(*list(model.children())[:-1]) if model_type != "affective" else nn.Sequential(*list(model.children())[:-2])
            model.eval()
            static_models.append(model)
        return static_models


    def __initialize_dynamic_model(self, model_type:str, path_to_weights:str)->nn.Module:
        if model_type in ["Dynamic_multi_modal_kinesics_facial_ER", "Dynamic_multi_modal_affective_kinesics_ER",
                          "Dynamic_multi_modal_affective_facial_ER"]:
            model = AttentionFusionModel_2dim(e1_num_features=256, e2_num_features=256, num_classes=3)
        elif model_type == "Dynamic_multi_modal_all_ER":
            model = AttentionFusionModel_3dim_v1(e1_num_features=256, e2_num_features=256, e3_num_features=256, num_classes=3)
        else:
            raise ValueError("Invalid model type. Only 'Dynamic_multi_modal_kinesics_facial_ER', "
                             "'Dynamic_multi_modal_affective_kinesics_ER', 'Dynamic_multi_modal_affective_facial_ER', "
                             "and 'Dynamic_multi_modal_all_ER' are supported for the dynamic model type.")
        model.load_state_dict(torch.load(path_to_weights))
        model.eval()
        return model


    def __preprocess_data(self, data:List[torch.Tensor])->List[torch.Tensor]:
        # every element of list has shape (batch_size, num_timesteps, channels, height, width)
        preprocessed_data = []
        for tmp_data, prep_funcs in zip(data, self.list_preprocessing_functions):
            prep_data = torch.stack([torch.stack([prep_funcs(image) for image in timestep]) for timestep in tmp_data])
            preprocessed_data.append(prep_data)
        return preprocessed_data


    def forward(self, x:List[torch.Tensor])->torch.Tensor:
        # x is a list of either two or three modalities, each with the shape (batch_size, num_timesteps, channels, height, width)
        # however, height and width can be different for different modalities
        # !!IMPORTANT: the order of the modalities is important. Here is the listed order for different combinations:
        #   2-modal: # affective - facial        # affective - kinesics     # kinesics - facial
        #   3-modal: # facial - kinesics - affective
        x = self.__preprocess_data(x)
        # pass the input through the static models
        x = [static_model(data) for static_model, data in zip(self.static_models, x)]
        # pass the input through the dynamic model
        result = self.dynamic_model(*x)
        return result