# Introduction
PyTorch implementation of three three approaches to knowledge distillation (Hinton et al.’s (2014), Romero et al.’s (2015) and Yim et al.’s (2017)) for comparison of student performance post-distillation on CIFAR-100 (Krizhevsky, 2009). Implementations of Hinton et al.'s and Yim et al.'s methods are modified from those presented in Hu's [repository](https://github.com/AberHu/Knowledge-Distillation-Zoo#readme) (Hu, 2022).

This repository also contains an implementation of dynamic weight average, a technique for automatically adjusting the weighting of different loss functions during training based on the rate of change of each loss function (Liu, et al. 2019), which can be optionally applied to Hinton and Romero's approaches. This implementation is modified from the code presented in Liu et al.'s [repository](https://github.com/lorenmt/mtan) (Liu, et al., 2022).

This repository utilises code for training procedures modified from implementations presented in the Dive into Deep Learning textbook (Zhang et al., 2021).

A [Jupyter notebook](distillation_interface.ipynb) is provided for running experiments. Titles and markdown are provided in the notebook for ease of use. We recommend using Google Colaboratory to run the notebook. The ResNet50 model used as a teacher in our experiments can be found [here](https://drive.google.com/file/d/1Vdlv-Aw1F0eEkkK89butpNDRFB1d1ozb/view?usp=sharing).

# Implementations
The values in the distiller_name column below can be used to access their respective knowledge distillation approaches in the Jupyter notebook. 

| Approach Name and Link                                            | Reference(s)                                | distiller_name  |
|:------------------------------------------------------------------|:--------------------------------------------|:----------------|
| [Logit-based knowledge distillation](distillation_methods_module/logits_distiller.py)                                | Hinton et al. (2014)                        | logits          |
| [Feature-based knowledge distillation](distillation_methods_module/features_distiller.py)                              | Romero et al. (2015)                        | features        |
| [Relation-based knowledge distillation](distillation_methods_module/relations_distiller.py)                             | Yim et al. (2017)                           | relations       |
| [Logit-based knowledge distillation with Dynamic Weight Average](distillation_methods_module/logits_distiller_dwa.py)    | Hinton et al. (2014) and Liu et al. (2019)  | logits-DWA      |
| [Feature-based knowledge distillation with Dynamic Weight Average](distillation_methods_module/features_distiller_dwa.py)  | Romero et al. (2015) and Liu et al. (2019)  | features-DWA    |

# References
Hinton, G., Vinyals, O. and Dean, J. (2014) ‘Distilling the Knowledge in a Neural Network’.
arXiv. Available at: http://arxiv.org/abs/1503.02531 (Accessed: 5 July 2022).

Hu, A. (2022) ‘Knowledge-Distillation-Zoo’. Available at: https://github.com/AberHu/Knowledge-Distillation-Zoo (Accessed: 30 October 2022).

Krizhevsky, A. (2009) Learning Multiple Layers of Features from Tiny Images. University of Toronto.

Romero, A., Ballas, N., Kahou, S.E., Chassang, A., Gatta, C. and Bengio, Y. (2015) ‘FitNets: Hints for Thin Deep Nets’.
arXiv. Available at: http://arxiv.org/abs/1412.6550 (Accessed: 31 August 2022).

Liu, S., Johns, E. and Davison, A.J. (2019) ‘End-to-End Multi-Task Learning with Attention’.
arXiv. Available at: http://arxiv.org/abs/1803.10704 (Accessed: 31 October 2022).

Liu, S., Johns, E. and Davison, A.J. (2022) ‘mtan/utils.py at master · lorenmt/mtan’.
Imperial College London. Available at: https://github.com/lorenmt/mtan (Accessed: 20 November 2022).

Yim, J., Joo, D., Bae, J. and Kim, J. (2017) ‘A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning’,
in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI: IEEE, pp. 7130–7138. Available at: https://doi.org/10.1109/CVPR.2017.754 (Accessed: 20 November 2022).

Zhang, A., Lipton, Z.C., Li, M. and Smola, A.J. (2021) Dive into Deep Learning. Available at: https://d2l.ai/ (Accessed: 20 November 2022).
