# Cross-Multi-Modal-Fusion-Approach-for-Enhanced-Engagement-Recognition
The official repository for the paper "A Cross-Multi-Modal Fusion Approach for Enhanced Engagement Recognition"


The weights for all models (static, uni-modal dynamic, multi-modal) can be accessed via this [link](https://drive.google.com/drive/folders/1VmQGsmD-cmEqYnVFpowxHlo_wyfuChwm?usp=sharing).


To see how to initialize, load, and exploit those models in your research, please, refer to [usage_example.py](https://github.com/icmi-2024-cross-multi-modal/icmi-cross-multi-modal/blob/main/usage_example.py).
Due to the constraints of GitHub, it is not possible for several models needed for the usage_example to be uploaded on GitHub. Therefore, please, download [this .zip file](https://drive.google.com/file/d/1OtWA4qxN40IXaXbnUSruCbBHObWVk-hg/view?usp=drive_link) and extract it into the *simpleHRNet* directory.

Also, to see all the details about the utilized Affective static model that has been trained on 10 different emotion recognition datasets, please, refer to [this directory](https://github.com/icmi-2024-cross-multi-modal/icmi-cross-multi-modal/tree/main/emotion_recognition).


The pipelines of all models are presented below.


<h3 align="center"> Multi-modal pipeline </h3>
<img alt="" align="center" src="https://github.com/icmi-2024-cross-multi-modal/icmi-cross-multi-modal/blob/main/figures/multimodal_pipeline.png" />

<h3 align="center"> Uni-modal dynamic models </h3>
<img alt="" align="center" src="https://github.com/icmi-2024-cross-multi-modal/icmi-cross-multi-modal/blob/main/figures/uni_modal_dynamic_pipeline.png" />

<h3 align="center"> Static Facial and Affective models </h3>
<img alt="" align="center" src="https://github.com/icmi-2024-cross-multi-modal/icmi-cross-multi-modal/blob/main/figures/facial_pipeline.png" />

<h3 align="center"> Static Kinesics model </h3>
<img alt="" align="center" src="https://github.com/icmi-2024-cross-multi-modal/icmi-cross-multi-modal/blob/main/figures/Kinesics_pipeline.png" />


<h3 align="center"> Hypertraining parameters </h3>
Also, to train both static and dynamic models, the following training hyperparameters were used:
AdamW optimizer with the learning rate set to 0.005 and weight decay to 0.0001.
Linear LR warmup with a starting value of 0.00001 for the first 100 training steps to avoid the gradient explosion. 
Cyclic LR scheduler with a minimum LR value of 0.0001 and an annealing period of 5. 
We set the early stopping number of epochs to 10 epochs. 

During the training of the static models, we have also used fine-tuning techniques known as gradual unfreezing and discriminative learning.
Discriminative learning is a training technique that fine-tunes pre-trained neural networks by setting different learning rates for each layer, helping to increase model performance on specific tasks. The main idea is that earlier layers extract more general features and should undergo minimal changes, while deeper layers are more task-specific and require significant adjustments, so learning rates are gradually decreased from deep to early layers. We applied a 0.85 factor for every LR of consecutive layers starting from newly initialized ones. On the other hand, gradual unfreezing is a technique that unfreezes DNN layers over training epochs to prevent overfitting or knowledge loss due to large initial gradients. Typically, only the added layers are unfrozen first, with more layers (often in blocks) being unfrozen in subsequent epochs, allowing the model to adjust gradually to the target task. We should note that we tried all possible combinations of those techniques for every static ER model. 




