from functools import partial

import torch
from fastai.layers import TimeDistributed
from torch import nn


from pytorch_utils.data_preprocessing import convert_image_to_float_and_scale
from pytorch_utils.layers.attention_layers import Transformer_layer
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
from pytorch_utils.models.Pose_estimation.HRNet import Modified_HRNet
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor
from torchvision import transforms
def get_emo_model_efficientNet_b1(path_to_weights:str):
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8, num_regression_neurons=2)
    model.load_state_dict(torch.load(path_to_weights))
    return model


def get_emo_embeddings_extractor_efficientNet_b1(path_to_weights:str):
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8, num_regression_neurons=2)
    model.load_state_dict(torch.load(path_to_weights))
    # cut off last two layers responsible for classification and regression
    model = torch.nn.Sequential(*list(model.children())[:-2])
    return model


def get_engagement_model_efficientNet_b1(path_to_weights:str):
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=3, num_regression_neurons=None)
    model.load_state_dict(torch.load(path_to_weights))
    return model


def get_engagement_model_HRNet(path_to_weights:str, path_to_HRNet_weights:str):
    model = Modified_HRNet(pretrained=True,
                               path_to_weights=path_to_HRNet_weights,
                               embeddings_layer_neurons=256, num_classes=8,
                               num_regression_neurons=None,
                               consider_only_upper_body=True)
    model.load_state_dict(torch.load(path_to_weights))
    return model


def get_preprocessing_functions(model_type:str):
    if model_type == 'EfficientNet-B1':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    elif model_type == 'EfficientNet-B4':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=380),
                                   EfficientNet_image_preprocessor()]
    elif model_type == 'Modified_HRNet':
        preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=256),
                                   convert_image_to_float_and_scale,
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]  # From HRNet
    return preprocessing_functions




class uni_modal_dynamic_facial_model(nn.Module):

    def __init__(self, facial_model:nn.Module, embeddings_layer_neurons:int,
                 num_classes:int, transformer_num_heads:int, num_timesteps:int):
        """ Creates a dynamic uni-modal facial model that takes as an input sequence of facial images
        and outputs a single engagement label for those images. The model have been trained on fixed number of timesteps (see paper).


        :param facial_model: torch.nn.Module
            The static facial engagement recognition model that can be obtained by get_engagement_model_efficientNet_b1 function.
        :param embeddings_layer_neurons: int
            The number of neurons in the embeddings layer of the model. (256)
        :param num_classes: int
            The number of classes in the classification task. (3)
        :param transformer_num_heads: int
            The number of heads in the transformer layer. (16)
        :param num_timesteps: int
            The number of timesteps in the input sequence. (depending on the model, usually 20)
        """
        super(uni_modal_dynamic_facial_model, self).__init__()
        self.embeddings_layer_neurons = embeddings_layer_neurons
        self.num_classes = num_classes
        self.transformer_num_heads = transformer_num_heads
        self.num_timesteps = num_timesteps

        # create facial model
        self.facial_model = TimeDistributed(facial_model)

        # create transformer layer for multimodal cross-fusion
        self.transformer_layer_1 = Transformer_layer(input_dim = embeddings_layer_neurons,
                                              num_heads=transformer_num_heads,
                                              dropout=0.2,
                                              positional_encoding=True)

        self.transformer_layer_2 = Transformer_layer(input_dim=embeddings_layer_neurons,
                                                num_heads=transformer_num_heads,
                                                dropout=0.2,
                                                positional_encoding=True)

        # get rid of timesteps
        self.squeeze_layer_1 = nn.Conv1d(num_timesteps, 1, 1)
        self.squeeze_layer_2 = nn.Linear(embeddings_layer_neurons, embeddings_layer_neurons//2)
        self.batch_norm = nn.BatchNorm1d(embeddings_layer_neurons//2)
        self.activation_squeeze_layer = nn.Tanh()
        self.end_dropout = nn.Dropout(0.2)

        # create classifier
        self.classifier = nn.Linear(embeddings_layer_neurons//2, num_classes)

    def forward(self, x):
        # facial model
        x = self.facial_model(x)
        # fusion
        x = self.transformer_layer_1(key=x, value=x, query=x)
        x = self.transformer_layer_2(key=x, value=x, query=x)
        # squeeze timesteps so that we have [batch_size, num_features]
        x = self.squeeze_layer_1(x)
        x = x.squeeze()
        # one more linear layer
        x = self.squeeze_layer_2(x)
        x = self.batch_norm(x)
        x = self.activation_squeeze_layer(x)
        x = self.end_dropout(x)
        # classifier
        x_classifier = self.classifier(x)

        return x_classifier



class uni_modal_dynamic_kinesics_model(nn.Module):

    def __init__(self, pose_model:nn.Module, embeddings_layer_neurons:int,
                 num_classes:int, transformer_num_heads:int, num_timesteps:int):
        """
        Creates a dynamic uni-modal kinesics model that takes as an input sequence of images with human bodies (including head)
        and outputs a single engagement label for those poses. The model have been trained on fixed number of timesteps (see paper).
        :param pose_model: torch.nn.Module
            The static kinesics engagement recognition model that can be obtained by get_engagement_model_HRNet function.
        :param embeddings_layer_neurons: int
            The number of neurons in the embeddings layer of the model. (256)
        :param num_classes: int
            The number of classes in the classification task. (3)
        :param transformer_num_heads: int
            The number of heads in the transformer layer. (16)
        :param num_timesteps: int
            The number of timesteps in the input sequence. (depending on the model, usually 20)
        """
        super(uni_modal_dynamic_kinesics_model, self).__init__()
        self.embeddings_layer_neurons = embeddings_layer_neurons
        self.num_classes = num_classes
        self.transformer_num_heads = transformer_num_heads
        self.num_timesteps = num_timesteps

        # create facial model
        self.pose_model = TimeDistributed(pose_model)

        # create transformer layer for multimodal cross-fusion
        self.transformer_layer_1 = Transformer_layer(input_dim = embeddings_layer_neurons,
                                              num_heads=transformer_num_heads,
                                              dropout=0.2,
                                              positional_encoding=True)

        self.transformer_layer_2 = Transformer_layer(input_dim=embeddings_layer_neurons,
                                                num_heads=transformer_num_heads,
                                                dropout=0.2,
                                                positional_encoding=True)

        # get rid of timesteps
        self.squeeze_layer_1 = nn.Conv1d(num_timesteps, 1, 1)
        self.squeeze_layer_2 = nn.Linear(embeddings_layer_neurons, embeddings_layer_neurons//2)
        self.batch_norm = nn.BatchNorm1d(embeddings_layer_neurons//2)
        self.activation_squeeze_layer = nn.Tanh()
        self.end_dropout = nn.Dropout(0.2)

        # create classifier
        self.classifier = nn.Linear(embeddings_layer_neurons//2, num_classes)

    def forward(self, x):
        # HRNet model
        x = self.pose_model(x)
        # fusion
        x = self.transformer_layer_1(key=x, value=x, query=x)
        x = self.transformer_layer_2(key=x, value=x, query=x)
        # squeeze timesteps so that we have [batch_size, num_features]
        x = self.squeeze_layer_1(x)
        x = x.squeeze()
        # one more linear layer
        x = self.squeeze_layer_2(x)
        x = self.batch_norm(x)
        x = self.activation_squeeze_layer(x)
        x = self.end_dropout(x)
        # classifier
        x_classifier = self.classifier(x)

        return x_classifier


class AttentionFusionModel_2dim(torch.nn.Module):

    def __init__(self, e1_num_features: int, e2_num_features: int, num_classes: int):
        # build cross-attention layers
        super(AttentionFusionModel_2dim, self).__init__()
        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.num_classes = num_classes
        self._build_cross_attention_modules(e1_num_features, e2_num_features)

        # build last fully connected layers
        self.classifier = torch.nn.Linear(128, num_classes)

    def _build_cross_attention_modules(self, e1_num_features: int, e2_num_features: int):
        self.e1_cross_att_layer_1 = Transformer_layer(input_dim=e1_num_features, num_heads=e1_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.e2_cross_att_layer_1 = Transformer_layer(input_dim=e2_num_features, num_heads=e2_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.e1_cross_att_layer_2 = Transformer_layer(input_dim=e1_num_features, num_heads=e1_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.e2_cross_att_layer_2 = Transformer_layer(input_dim=e2_num_features, num_heads=e2_num_features // 4,
                                                      dropout=0.1, positional_encoding=True)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_ett_dense_layer = torch.nn.Linear((e1_num_features + e2_num_features) * 2, 128)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(128)
        self.cross_att_activation = torch.nn.ReLU()

    def forward_cross_attention(self, e1, e2):
        # cross attention 1
        e1 = self.e1_cross_att_layer_1(key=e1, value=e1,
                                       query=e2)  # Output shape (batch_size, sequence_length, e1_num_features)
        e2 = self.e2_cross_att_layer_1(key=e2, value=e2,
                                       query=e1)  # Output shape (batch_size, sequence_length, e2_num_features)
        # cross attention 2
        e1 = self.e1_cross_att_layer_2(key=e1, value=e1,
                                       query=e2)  # Output shape (batch_size, sequence_length, e1_num_features)
        e2 = self.e2_cross_att_layer_2(key=e2, value=e2,
                                       query=e1)  # Output shape (batch_size, sequence_length, e2_num_features)
        # concat e1 and e2
        x = torch.cat((e1, e2),
                      dim=-1)  # Output shape (batch_size, sequence_length, num_features = e1_num_features + e2_num_features)
        # permute it to (batch_size, num_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        x = x.permute(0, 2, 1)  # Output size (batch_size, num_features, sequence_length)
        # calculate 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(dim=-1)  # Output shape (batch_size, num_features)
        max_pool = max_pool.squeeze(dim=-1)  # Output shape (batch_size, num_features)
        # concat avg_pool and max_pool
        x = torch.cat((avg_pool, max_pool), dim=-1)  # Output shape (batch_size, num_features*2)
        # dense layer
        x = self.cross_ett_dense_layer(x)
        x = self.cross_att_batch_norm(x)
        x = self.cross_att_activation(x)
        return x

    def forward(self, feature_set_1, feature_set_2):
        e1, e2 = feature_set_1, feature_set_2
        # cross attention
        x = self.forward_cross_attention(e1, e2)
        # classifier
        x = self.classifier(x)
        return x


class AttentionFusionModel_3dim_v1(torch.nn.Module):

    def __init__(self, e1_num_features: int, e2_num_features: int, e3_num_features: int, num_classes: int):
        # check if all the embeddings have the same sequence length
        super().__init__()
        assert e1_num_features == e2_num_features == e3_num_features

        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.e3_num_features = e3_num_features
        self.num_classes = num_classes
        # create cross-attention blocks
        self.__build_cross_attention_blocks()
        # build last fully connected layers
        self.classifier = torch.nn.Linear(256, num_classes)

    def __build_cross_attention_blocks(self):
        # assuming that the embeddings are ordered as 'facial', 'pose', 'affective', we need to build several cross-attention blocks
        # first block will be the block, where the facial modality is the main one, while pose or affective are the secondary ones
        # f - facial, p - pose, a - affective.  THen f_p - facial is the main modality, pose is the secondary one,
        # f_a - facial is the main modality, affective is the secondary one, and so on
        # main - facial
        self.block_f_p = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        self.block_f_a = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        # main - pose
        self.block_p_f = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        self.block_p_a = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        # main - affective
        self.block_a_f = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)
        self.block_a_p = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                           positional_encoding=True)

        # now we need to do a cross attention with the remaining modality. For example, for the f_p block it will be
        # cross attention with the affective modelity. As a result, we get f_p_a block
        # for f_a block it will be cross attention with the pose modality. As a result, we get f_a_p block, and so on.
        # main - facial
        self.block_f_p_a = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        self.block_f_a_p = Transformer_layer(input_dim=self.e1_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        # main - pose
        self.block_p_f_a = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        self.block_p_a_f = Transformer_layer(input_dim=self.e2_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        # main - affective
        self.block_a_f_p = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)
        self.block_a_p_f = Transformer_layer(input_dim=self.e3_num_features, num_heads=8, dropout=0.1,
                                             positional_encoding=True)

        # at the end of the cross attention we want to calculate 1D avg pooling and 1D max pooling to 'aggregate'
        # the temporal information
        # at the time of the pooling, embeddings from all different cross-attention blocks will be concatenated
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_att_dense_layer = torch.nn.Linear(
            (self.e1_num_features + self.e2_num_features + self.e3_num_features) * 2*2, 256)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(256)
        self.cross_att_activation = torch.nn.ReLU()

    def __forward_cross_attention_first_stage(self, f, p, a):
        # f - facial, p - pose, a - affective
        # main - facial
        f_p = self.block_f_p(key = p, value = p, query = f)
        f_a = self.block_f_a(key = a, value = a, query = f)
        # main - pose
        p_f = self.block_p_f(key = f, value = f, query = p)
        p_a = self.block_p_a(key = a, value = a, query = p)
        # main - affective
        a_f = self.block_a_f(key = f, value = f, query = a)
        a_p = self.block_a_p(key = p, value = p, query = a)
        return f_p, f_a, p_f, p_a, a_f, a_p

    def __forward_cross_attention_second_stage(self, f_p, f_a, p_f, p_a, a_f, a_p, f, p, a):
        # f - facial, p - pose, a - affective
        # main - facial
        f_p_a = self.block_f_p_a(key = a, value = a, query = f_p)
        f_a_p = self.block_f_a_p(key = p, value = p, query = f_a)
        # main - pose
        p_f_a = self.block_p_f_a(key = a, value = a, query = p_f)
        p_a_f = self.block_p_a_f(key = f, value = f, query = p_a)
        # main - affective
        a_f_p = self.block_a_f_p(key = p, value = p, query = a_f)
        a_p_f = self.block_a_p_f(key = f, value = f, query = a_p)
        return f_p_a, f_a_p, p_f_a, p_a_f, a_f_p, a_p_f

    def forward(self, f, p, a):
        # cross attention first stage
        f_p, f_a, p_f, p_a, a_f, a_p = self.__forward_cross_attention_first_stage(f, p, a)
        # cross attention second stage
        f_p_a, f_a_p, p_f_a, p_a_f, a_f_p, a_p_f = self.__forward_cross_attention_second_stage(f_p, f_a,
                                                                                               p_f, p_a,
                                                                                               a_f, a_p,
                                                                                               f, p, a)
        # now we need to concatenate all the embeddings from the second stage
        # concat
        output = torch.cat((f_p_a, f_a_p, p_f_a, p_a_f, a_f_p, a_p_f),
                           dim=-1)  # Output shape (batch_size, sequence_length, num_features = e1_num_features + e2_num_features + e3_num_features)
        # permute it to (batch_size, num_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        output = output.permute(0, 2, 1)  # Output size (batch_size, num_features, sequence_length)
        # 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(output)  # Output size (batch_size, num_features, 1)
        max_pool = self.max_pool(output)  # Output size (batch_size, num_features, 1)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(-1)  # Output size (batch_size, num_features)
        max_pool = max_pool.squeeze(-1)  # Output size (batch_size, num_features)
        # concat avg_pool and max_pool
        output = torch.cat((avg_pool, max_pool), dim=1)  # Output size (batch_size, num_features * 2)
        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(output)  # Output size (batch_size, 256)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        # last classification layer
        output = self.classifier(output)  # Output size (batch_size, num_classes)
        return output