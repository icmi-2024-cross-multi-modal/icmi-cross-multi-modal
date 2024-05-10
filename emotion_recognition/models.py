import torch

from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1


def load_emo_embeddings_extractor_model(weights_path: str) -> torch.nn.Module:
    """ Creates and loads the emotion recognition model for embeddings extraction. To do so, the cutting off some last
    layers of the model is required.
    """
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8, num_regression_neurons=2)
    model.load_state_dict(torch.load(weights_path))
    # cut off last two layers responsible for classification and regression
    model = torch.nn.Sequential(*list(model.children())[:-2])
    return model




def load_emo_model(weights_path: str) -> torch.nn.Module:
    """ Creates and loads the emotion recognition model based on EfficientNet B1 architecture.
    Take into account that the model outputs the tuple of two values: classification (array with 8 class probabilities)
    and regression (2 - arousal, valence)."""
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8, num_regression_neurons=2)
    model.load_state_dict(torch.load(weights_path))
    return model