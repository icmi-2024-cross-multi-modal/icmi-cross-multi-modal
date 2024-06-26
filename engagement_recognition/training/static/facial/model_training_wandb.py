import argparse
import gc
import os
from typing import Optional

import torch
import wandb
from torchinfo import summary

from engagement_recognition.training.static import training_config
from engagement_recognition.training.static.data_preparation import load_data_and_construct_dataloaders
from engagement_recognition.training.static.model_evaluation import evaluate_model
from pytorch_utils.lr_schedullers import WarmUpScheduler
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, \
    Modified_EfficientNet_B4
from pytorch_utils.training_utils.callbacks import TorchEarlyStopping, GradualLayersUnfreezer, gradually_decrease_lr
from pytorch_utils.training_utils.general_functions import train_epoch
from pytorch_utils.training_utils.losses import SoftFocalLoss



def train_model(train_generator: torch.utils.data.DataLoader, dev_generator: torch.utils.data.DataLoader,
                class_weights: torch.Tensor, MODEL_TYPE: str, BATCH_SIZE: int, ACCUMULATE_GRADIENTS: int,
                GRADUAL_UNFREEZING: Optional[bool] = False,
                DISCRIMINATIVE_LEARNING: Optional[bool] = False,
                loss_multiplication_factor: Optional[float] = None) -> None:
    print("Start of the model training. Gradual_unfreezing:%s, Discriminative_lr:%s" % (GRADUAL_UNFREEZING,
                                                                                        DISCRIMINATIVE_LEARNING))
    # metaparams
    metaparams = {
        # general params
        "architecture": MODEL_TYPE,
        "MODEL_TYPE": MODEL_TYPE,
        "dataset": "NoXi, DAiSEE",
        "BEST_MODEL_SAVE_PATH": training_config.BEST_MODEL_SAVE_PATH,
        "NUM_WORKERS": training_config.NUM_WORKERS,
        # model architecture
        "NUM_CLASSES": training_config.NUM_CLASSES,
        # training metaparams
        "NUM_EPOCHS": training_config.NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "OPTIMIZER": training_config.OPTIMIZER,
        "AUGMENT_PROB": training_config.AUGMENT_PROB,
        "EARLY_STOPPING_PATIENCE": training_config.EARLY_STOPPING_PATIENCE,
        "WEIGHT_DECAY": training_config.WEIGHT_DECAY,
        # LR scheduller params
        "LR_SCHEDULLER": training_config.LR_SCHEDULLER,
        "ANNEALING_PERIOD": training_config.ANNEALING_PERIOD,
        "LR_MAX_CYCLIC": training_config.LR_MAX_CYCLIC,
        "LR_MIN_CYCLIC": training_config.LR_MIN_CYCLIC,
        "LR_MIN_WARMUP": training_config.LR_MIN_WARMUP,
        "WARMUP_STEPS": training_config.WARMUP_STEPS,
        "WARMUP_MODE": training_config.WARMUP_MODE,
        # gradual unfreezing (if applied)
        "GRADUAL_UNFREEZING": GRADUAL_UNFREEZING,
        "UNFREEZING_LAYERS_PER_EPOCH": training_config.UNFREEZING_LAYERS_PER_EPOCH,
        "LAYERS_TO_UNFREEZE_BEFORE_START": training_config.LAYERS_TO_UNFREEZE_BEFORE_START,
        # discriminative learning
        "DISCRIMINATIVE_LEARNING": DISCRIMINATIVE_LEARNING,
        "DISCRIMINATIVE_LEARNING_INITIAL_LR": training_config.DISCRIMINATIVE_LEARNING_INITIAL_LR,
        "DISCRIMINATIVE_LEARNING_MINIMAL_LR": training_config.DISCRIMINATIVE_LEARNING_MINIMAL_LR,
        "DISCRIMINATIVE_LEARNING_MULTIPLICATOR": training_config.DISCRIMINATIVE_LEARNING_MULTIPLICATOR,
        "DISCRIMINATIVE_LEARNING_STEP": training_config.DISCRIMINATIVE_LEARNING_STEP,
        "DISCRIMINATIVE_LEARNING_START_LAYER": training_config.DISCRIMINATIVE_LEARNING_START_LAYER,
        # loss params
        "loss_multiplication_factor": loss_multiplication_factor,
    }
    print("____________________________________________________")
    print("Training params:")
    for key, value in metaparams.items():
        print(f"{key}: {value}")
    print("____________________________________________________")
    # initialization of Weights and Biases
    wandb.init(project="Engagement_recognition_F2F", config=metaparams)
    config = wandb.config
    wandb.config.update({'BEST_MODEL_SAVE_PATH': wandb.run.dir}, allow_val_change=True)

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.MODEL_TYPE == "EfficientNet-B1":
        model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=config.NUM_CLASSES,
                                         num_regression_neurons=None)
    elif config.MODEL_TYPE == "EfficientNet-B4":
        model = Modified_EfficientNet_B4(embeddings_layer_neurons=256, num_classes=config.NUM_CLASSES,
                                         num_regression_neurons=None)
    else:
        raise ValueError("Unknown model type: %s" % config.MODEL_TYPE)
    model = model.to(device)
    # print model architecture
    summary(model, (2, 3, 380, 380))

    # define all model layers (params), which will be used by optimizer
    if config.MODEL_TYPE == "EfficientNet-B1" or config.MODEL_TYPE == "EfficientNet-B4":
        model_layers = [
            *list(list(model.children())[0].features.children()),
            *list(list(model.children())[0].children())[1:],
            *list(model.children())[1:]  # added layers
        ]
    else:
        raise ValueError("Unknown model type: %s" % config.MODEL_TYPE)
    # layers unfreezer
    if GRADUAL_UNFREEZING:
        layers_unfreezer = GradualLayersUnfreezer(model=model, layers=model_layers,
                                                  layers_per_epoch=config.UNFREEZING_LAYERS_PER_EPOCH,
                                                  layers_to_unfreeze_before_start=config.LAYERS_TO_UNFREEZE_BEFORE_START,
                                                  input_shape=(config.BATCH_SIZE, 3,
                                                               training_config.MODEL_INPUT_SIZE[config.MODEL_TYPE],
                                                               training_config.MODEL_INPUT_SIZE[config.MODEL_TYPE]),
                                                  verbose=True)
    # if discriminative learning is applied
    if DISCRIMINATIVE_LEARNING:
        model_parameters = gradually_decrease_lr(layers=model_layers,
                                                 initial_lr=config.DISCRIMINATIVE_LEARNING_INITIAL_LR,
                                                 multiplicator=config.DISCRIMINATIVE_LEARNING_MULTIPLICATOR,
                                                 minimal_lr=config.DISCRIMINATIVE_LEARNING_MINIMAL_LR,
                                                 step=config.DISCRIMINATIVE_LEARNING_STEP,
                                                 start_layer=config.DISCRIMINATIVE_LEARNING_START_LAYER)
        for param_group in model_parameters:
            print("size: {}, lr: {}".format(param_group['params'].shape, param_group['lr']))
        print(
            'The learning rate was changed for each layer according to discriminative learning approach. The new learning rates are:')
    else:
        model_parameters = model.parameters()
    # select optimizer
    optimizers = {'Adam': torch.optim.Adam,
                  'SGD': torch.optim.SGD,
                  'RMSprop': torch.optim.RMSprop,
                  'AdamW': torch.optim.AdamW}
    optimizer = optimizers[config.OPTIMIZER](model_parameters, lr=config.LR_MAX_CYCLIC,
                                             weight_decay=config.WEIGHT_DECAY)
    # Loss functions
    class_weights = class_weights.to(device)
    criterions = [SoftFocalLoss(softmax=True, alpha=class_weights, gamma=2)]
    # create LR scheduler
    lr_schedullers = {
        'Cyclic': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ANNEALING_PERIOD,
                                                             eta_min=config.LR_MIN_CYCLIC),
        'ReduceLRonPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8),
        'Warmup_cyclic': WarmUpScheduler(optimizer=optimizer,
                                         lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                                 T_max=config.ANNEALING_PERIOD,
                                                                                                 eta_min=config.LR_MIN_CYCLIC),
                                         len_loader=len(train_generator) // ACCUMULATE_GRADIENTS,
                                         warmup_steps=config.WARMUP_STEPS,
                                         warmup_start_lr=config.LR_MIN_WARMUP,
                                         warmup_mode=config.WARMUP_MODE)
    }
    # if we use discriminative learning, we don't need LR scheduler
    if DISCRIMINATIVE_LEARNING:
        lr_scheduller = None
    else:
        lr_scheduller = lr_schedullers[config.LR_SCHEDULLER]
        # if lr_scheduller is warmup_cyclic, we need to change the learning rate of optimizer
        if config.LR_SCHEDULLER == 'Warmup_cyclic':
            optimizer.param_groups[0]['lr'] = config.LR_MIN_WARMUP

    # early stopping
    best_val_recall_classification = 0
    early_stopping_callback = TorchEarlyStopping(verbose=True, patience=config.EARLY_STOPPING_PATIENCE,
                                                 save_path=config.BEST_MODEL_SAVE_PATH,
                                                 mode="max")

    # train model
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %i" % epoch)
        # train the model
        model.train()
        train_loss = train_epoch(model, train_generator, optimizer, criterions, device, print_step=100,
                                 accumulate_gradients=ACCUMULATE_GRADIENTS,
                                 batch_wise_lr_scheduller=lr_scheduller if config.LR_SCHEDULLER == 'Warmup_cyclic' else None,
                                 loss_multiplication_factor=config.loss_multiplication_factor)
        print("Train loss: %.10f" % train_loss)

        # validate the model
        model.eval()
        print("Evaluation of the model on dev set.")
        val_metrics_classification = evaluate_model(model, dev_generator, device, print_metrics=True,
                                                    metrics_name_prefix='val_')

        # update best val metrics got on validation set and log them using wandb
        # also, save model if we got better recall
        if val_metrics_classification['val_recall_classification'] > best_val_recall_classification:
            best_val_recall_classification = val_metrics_classification['val_recall_classification']
            wandb.config.update({'best_val_recall_classification': best_val_recall_classification},
                                allow_val_change=True)
            # save best model
            if not os.path.exists(config.BEST_MODEL_SAVE_PATH):
                os.makedirs(config.BEST_MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(config.BEST_MODEL_SAVE_PATH, 'best_model_recall.pth'))

        # log everything using wandb
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log({'learning_rate': optimizer.param_groups[0]["lr"]}, commit=False)
        wandb.log(val_metrics_classification, commit=False)
        wandb.log({'train_loss': train_loss})
        # update LR if needed
        if config.LR_SCHEDULLER == 'ReduceLRonPlateau':
            lr_scheduller.step(best_val_recall_classification)
        elif config.LR_SCHEDULLER == 'Cyclic':
            lr_scheduller.step()

        # check early stopping
        early_stopping_result = early_stopping_callback(best_val_recall_classification, model)
        if early_stopping_result:
            print("Early stopping")
            break
        # unfreeze next n layers
        if GRADUAL_UNFREEZING:
            layers_unfreezer()
    # clear RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(model_type, batch_size, accumulate_gradients, gradual_unfreezing, discriminative_learning,
         loss_multiplication_factor):
    print("Start of the script....")
    # get data loaders
    (train_generator, dev_generator, test_generator), class_weights = load_data_and_construct_dataloaders(
        path_to_data_NoXi=training_config.NOXI_FACIAL_PATH,
        path_to_data_DAiSEE=training_config.DAISEE_FACIAL_PATH,
        model_type=model_type,
        batch_size=batch_size,
        return_class_weights=True)
    # train the model
    train_model(train_generator=train_generator, dev_generator=dev_generator, class_weights=class_weights,
                MODEL_TYPE=model_type, BATCH_SIZE=batch_size, ACCUMULATE_GRADIENTS=accumulate_gradients,
                GRADUAL_UNFREEZING=gradual_unfreezing, DISCRIMINATIVE_LEARNING=discriminative_learning,
                loss_multiplication_factor=loss_multiplication_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Emotion Recognition model training',
        epilog='Parameters: model_type, batch_size, accumulate_gradients, gradual_unfreezing, discriminative_learning')
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--accumulate_gradients', type=int, required=True)
    parser.add_argument('--gradual_unfreezing', type=int, required=True)
    parser.add_argument('--discriminative_learning', type=int, required=True)
    parser.add_argument('--loss_multiplication_factor', type=float, required=False, default=1.0)
    args = parser.parse_args()
    # turn passed args from int to bool
    print("Passed args: ", args)
    # check arguments
    if args.model_type not in ['EfficientNet-B1', 'EfficientNet-B4']:
        raise ValueError("model_type should be either EfficientNet-B1 or EfficientNet-B4. Got %s" % args.model_type)
    if args.batch_size < 1:
        raise ValueError("batch_size should be greater than 0")
    if args.accumulate_gradients < 1:
        raise ValueError("accumulate_gradients should be greater than 0")
    if args.gradual_unfreezing not in [0, 1]:
        raise ValueError("gradual_unfreezing should be either 0 or 1")
    if args.discriminative_learning not in [0, 1]:
        raise ValueError("discriminative_learning should be either 0 or 1")
    # convert args to bool
    gradual_unfreezing = True if args.gradual_unfreezing == 1 else False
    discriminative_learning = True if args.discriminative_learning == 1 else False
    model_type = args.model_type
    batch_size = args.batch_size
    accumulate_gradients = args.accumulate_gradients
    loss_multiplication_factor = args.loss_multiplication_factor
    # run main script with passed args
    main(model_type=model_type, batch_size=batch_size, accumulate_gradients=accumulate_gradients,
         gradual_unfreezing=gradual_unfreezing,
         discriminative_learning=discriminative_learning,
         loss_multiplication_factor=loss_multiplication_factor)
    # clear RAM
    gc.collect()
    torch.cuda.empty_cache()
