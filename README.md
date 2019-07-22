
# Knowledge distillation

  

This repo contains a simple example of knowledge distillation between two neural networks, written in Tensorflow low-level API. Teacher networks accomplish accuracy of 69% on the test set. After distillation of knowledge, student network accomplishes accuracy of 68% on the test set with 92% fewer parameters.

  
## Papers

* [Distilling the Knowledge in a Neural Network, Geoffrey Hinton, Oriol Vinyals, Jeff Dean](https://arxiv.org/abs/1503.02531)
* [Model Distillation with Knowledge Transfer from Face Classification to Alignment and Verification, Chong Wang, Xipeng Lan, Yangang Zhang](https://arxiv.org/abs/1709.02929)
* [FitNets: Hints for Thin Deep Nets, Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio](https://arxiv.org/abs/1412.6550)

## Dataset

* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)


## Requirements

* Tensorflow 1.13.1

* Numpy 1.16.2

## Future work

* increase accuracy of both teacher and student network
* implement hints learning explained in FitNets paper
