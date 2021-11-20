# Trustworthy_AI
# CIFAR10 with PyTorch

## Environment

python  --  3+

pytorch --  1.0+

## Dataset

CIFAR10

## Train & Test

python main.py 

--lr: learning rate

--optim: the optimizer for model training

--ep:  the epoch for model training

--seed: the random seed

## Evaluation

<center class="half">    
  <img style="border-radius: 0.3125em;"     src="figure\ada_loss.png" width="400">    
  <img style="border-radius: 0.3125em;"     src="figure\ada_acc.png" width="400">    
  <br>    
</center>
图1 Adagrad的loss和acc，其中loss for Adagrad = 0.47054, acc for Adagrad = 0.85560


<center class="half">    
  <img style="border-radius: 0.3125em;"     src="figure\adam_loss.png" width="400">    
  <img style="border-radius: 0.3125em;"     src="figure\adam_acc.png" width="400">    
</center>
图2 Adam的loss和acc, 其中loss for Adam= 0.37655, acc for for Adam = 0.87870


<center class="half">    
  <img style="border-radius: 0.3125em;"     src="figure\sgd_loss.png" width="400">    
  <img style="border-radius: 0.3125em;"     src="figure\sgd_acc.png" width="400">    
  <br>    
</center>
图3 Sgd的loss和acc，其中loss for Sgd = 0.40721, acc for Sgd = 0.88750

<center class="half"> </center>  
表1 实验结果对比
| Optimizer |     Loss     | Acc       |
| --------- |--------------| ----------|
| Adagrad   |     0.4705   |    0.8556 |
| Adam      |     0.3766   |    0.8787 |
| SGD       |     0.4072   |    0.8875 |

在本次实验中，我选择了三个不同的优化器对模型进行训练，分别为adagrad、adam和sgd。我们不难发现，不同的优化器类型对模型训练的影响十分大。

从结果角度出发，SGD的效果是最好的，在50个epoch的前提下，acc可以达到0.8875，其次是Adam的0.8787，最差的是adagrad的0.8556。我们对比曲线图可以发现，adam在训练集收敛的是最快的，测试集上的loss也比较低，但它在valid集上的loss和acc的波动也是比较大的，这说明了该优化器的不稳定性。相比之下，sgd和adagrad的波动会相对平稳不少，但是收敛得会比较慢。此外，adagrad相比sgd收敛较慢，且有过拟合的倾向。

综上，在现有的前提条件的情况下中SGD的表现最为优越。然而在实际问题中，即使是优化器一定的情况下，优化器本身还有许多值得调整的参数，因此该实验只能从宏观角度进行解释。


