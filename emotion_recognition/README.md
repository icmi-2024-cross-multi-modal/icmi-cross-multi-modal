The Affective static model that has been used as feature extractor of affective features. Those features have significantly improved the engagement recognition performance. 

The weights for the best affective model can be accessed via this [link](https://drive.google.com/drive/folders/16nL_5J9I9MEnJQI34fgRMdSUm9HwfgwX?usp=drive_link).


<h3 align="center"> The pipeline of the model </h3>
<img alt="" align="center" src="https://github.com/ECCV-2024-cross-multi-modal/Cross-Multi-Modal-Fusion-Approach/blob/main/figures/facial_pipeline.png" />

<h3 align="center">Training Data</h3>

To train, validate, and test the static emotion recognition model, we have used several publicly available datasets:
1. RECOLA [1]
2. SEWA [2]
3. SEMAINE [3]
4. AFEW-VA [4]
5. AffectNet [5]
6. SAVEE [6,7,8]
7. EMOTIC [9]
8. ExpW [10,11]
9. FER+ [12]
10. RAF-DB [13,14]

Both, the categorical and Valence-Arousal annotations have been used during the training.

<h3 align="center">Training Hyperparameters</h3>
To train the affective static emotion recognition model based on EfficientNet-B1 neural network architecture, the following parameters have been used:
AdamW optimizer with the learning rate set to 0.005 and weight decay to 0.0001.
Linear LR warmup with a starting value of 0.00001 for the first 100 training steps to avoid the gradient explosion. 
Cyclic LR scheduler with a minimum LR value of 0.0001 and an annealing period of 5. 
We set the early stopping number of epochs to 30 epochs. 

During the training, we have also used fine-tuning techniques known as *gradual unfreezing* and *discriminative learning*.
*Discriminative learning* is a training technique that fine-tunes pre-trained neural networks by setting different learning rates for each layer, helping to increase model performance on specific tasks. The main idea is that earlier layers extract more general features and should undergo minimal changes, while deeper layers are more task-specific and require significant adjustments, so learning rates are gradually decreased from deep to early layers. We applied a 0.85 factor for every LR of consecutive layers starting from newly initialized ones. On the other hand, *gradual unfreezing* is a technique that unfreezes DNN layers over training epochs to prevent overfitting or knowledge loss due to large initial gradients. Typically, only the added layers are unfrozen first, with more layers (often in blocks) being unfrozen in subsequent epochs, allowing the model to adjust gradually to the target task. We should note that we tried all possible combinations of those techniques for every static ER model. 


<h3>Literature</h3>
[1] Ringeval, Fabien, et al. "Introducing the RECOLA multimodal corpus of remote collaborative and affective interactions." 2013 10th IEEE international conference and workshops on automatic face and gesture recognition (FG). IEEE, 2013.<br />
[2] Kossaifi, Jean, et al. "Sewa db: A rich database for audio-visual emotion and sentiment research in the wild." IEEE transactions on pattern analysis and machine intelligence 43.3 (2019): 1022-1040.<br />
[3] McKeown, Gary, et al. "The semaine database: Annotated multimodal records of emotionally colored conversations between a person and a limited agent." IEEE transactions on affective computing 3.1 (2011): 5-17.<br />
[4] Kossaifi, Jean, et al. "AFEW-VA database for valence and arousal estimation in-the-wild." Image and Vision Computing 65 (2017): 23-36.<br />
[5] Mollahosseini, Ali, Behzad Hasani, and Mohammad H. Mahoor. "Affectnet: A database for facial expression, valence, and arousal computing in the wild." IEEE Transactions on Affective Computing 10.1 (2017): 18-31.<br />
[6] S. Haq, P.J.B. Jackson, and J.D. Edge. Audio-Visual Feature Selection and Reduction for Emotion Classification. In Proc. Int'l Conf. on Auditory-Visual Speech Processing, pages 185-190, 2008. <br />
[7] S. Haq and P.J.B. Jackson. "Speaker-Dependent Audio-Visual Emotion Recognition", In Proc. Int'l Conf. on Auditory-Visual Speech Processing, pages 53-58, 2009.<br />
[8] S. Haq and P.J.B. Jackson, "Multimodal Emotion Recognition", In W. Wang (ed), Machine Audition: Principles, Algorithms and Systems, IGI Global Press, ISBN 978-1615209194, chapter 17, pp. 398-423, 2010.<br />
[9] Kosti, Ronak, et al. "Emotic: Emotions in context dataset." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.<br />
[10] Zhang, Zhanpeng, et al. "Learning social relation traits from face images." Proceedings of the IEEE international conference on computer vision. 2015.<br />
[11] Zhang, Zhanpeng, et al. "From facial expression recognition to interpersonal relation prediction." International Journal of Computer Vision 126 (2018): 550-569.<br />
[12] Barsoum, Emad, et al. "Training deep networks for facial expression recognition with crowd-sourced label distribution." Proceedings of the 18th ACM international conference on multimodal interaction. 2016.<br />
[13] Li, Shan, Weihong Deng, and JunPing Du. "Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.<br />
[14] Shan, Li, and Weihong Deng. "Reliable crowdsourcing and deep locality-preserving learning for unconstrained facial expression recognition." IEEE Transactions on Image Processing 28.1 (2018): 356-370.<br />

