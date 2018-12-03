# 第14章生成对抗网络

生成对抗网络（GANs）是一种深层神经网络结构，它由两个相互竞争的网络组成（这也是将其称之为对抗网络的原因。）
2014年，蒙特利尔大学(University of Montreal)的伊恩•古德费勒(Ian Goodfellow)和包括约舒亚•本吉奥(yoshu Bengio)在内的研究人员在一篇论文
(https://arxiv.org/abs/1406.2661)中介绍了GANs。说到GANs，脸书的人工智能研究主管，扬·勒恩（Yann LeCun）称，这是在过去十年的机器学习中最
有趣的想法。GANs的潜力是巨大的，因为它们可以学习模拟任何数据分布。也就是说，GANs可以在任何领域(图像、音乐、演讲或散文)创建与我们自己的世界
极其相似的世界。从某种意义上说，它们是机器人艺术家，它们的作品令人印象深刻。(http://www.nytimes.com/2017/08/14/arts/design/google-how-ai-create 
-new music and new-artists-project-magenta.html)。

本章将讨论以下内容:
### 对GANs的一个直观介绍
### GANs的简单实现
### 深度卷积生成式对抗网络

## 对GANs的一个直观介绍

在本节中，我们将以一种非常直观的方式介绍GANs。为了了解GANs是如何工作的，将模拟一个获得门票的情景。
故事开始于一个非常有趣的活动，而你想参加这个活动。但是你听说这个活动时所有的票都卖完了，你还是要想尽办法去参加它。所以你想到了一个主意!你想自制一张
与正真的门票完全相同或非常相似的票。这并不容易，一个挑战就是你不知道正真的门票是什么样子的，只能根据自己的想象来设计这张票。你要先设计好门票，然后到
活动现场，把门票给保安看。希望他们能让你进去。但是你并不想引起保安的注意，所以你决定求助于你的朋友，他会带你先设计出来的票给保安看。如果未通过，他会
观察进去的人拿着的正真的票的样子，给你一些相关信息。你将根据这些信息来修改门票，直到保安允许他进入。以此类推，你将设计一张能让自己通过的门票。不管这
个故事有多假，GANs的工作方式确实跟它很相近。现今GANs非常流行，人们在计算机视觉领域的许多应用中都使用它。在许多有趣的应用中都用到了GANs，在此我们
将提到并实现其中一些。
GNAs包含两个在许多计算机视觉领域都已取得突破成果的主要组成部分。第一个组成部分称为生成器，第二称为判别器：生成器将尝试从特定的概率分布中生成数据样本，
这与试图复制活动门票的人非常相似；判别器将会判断（就像保安试图从票中发现瑕疵以此确定票是真还是假）是否它的输入来自初始训练集（门票）或者来自生成器部分
（即设计门票的人）

## GANs的简单实现

从伪造活动的门票来看，GNAs 的思想似乎是非常直观的。为了更加明白地理解GNAs 是如何工作的以及如何实现它，我们来展示一个GNAs基于MNIST 数据集的简单实现。
首先，我们要构建核心GANs网络，这由两个部分组成：生成器和判别器。正如我们说的，生成器将尝试从一个特定的概率分布中想象或伪造数据样本;判别器通过对实际
数据样本的访问和查看，判断出生成器的输出在设计上是否有缺陷，或者是否与原始数据样本非常接近。与活动情景类似，生成器的全部目的是试图使判别器相信生成的
图像来自真实数据集，从而试图欺骗判别器。训练过程与活动故事有相似的结局，生成器将最终生成与原始数据样本非常相似的图像:

任何GAN的典型结构如图2所示，并将在MNIST数据集上进行训练。图中Latent样本部分是一个随机的想法或向量，生成器使用它来用假图像复制真实图像。正如我们所提
到的，判别器作为一种判断器，它会试着将真实的图像从生成器设计的虚假图像中分离出来。因此，这个网络的输出将是二进制的，它可以用一个sigmoid函数表示，
该函数的值为0(表示输入是假图像)和1(表示输入是真实图像)。让我们开始实现这个结构，看看它如何在MNIST数据集上执行。

首先导入实现所需的库:
```%matplotlib inline
Import matplotlib.pyplot as plt
Import pickle as pkl
Import numpy as np
Import tensorflow as tf
```
我们将使用MNIST数据集，因此需要用用TensorFlow helper获取数据集并将其存储在某处:
```from tensorflow.example.tutorials.mnist import input_data
mnist_dataset = input_data.read_data_sets(‘MNIST_data’)
```
Output:
```Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
```
输入模型
在深入构建由生成器和判别器表示的GAN核心之前，先定义计算图的输入。如图2所示，我们需要两个输入。第一个是真实的图像，它将被输入判别器。另一种输入
称为潜在空间，它将被输入到生成器，并用于生成它的虚假图像:

 ```Defining the model input for the generator and discrimator
def inputs_placeholders (discrimator_real_dim, gen_z_dim):
   real_discrminator_input = tf.placeholder(tf.float32, (None,
discrimator_real_dim), name=”real_discrminator_input” )
       generator_inputs_z = tf.placeholder(tf.float32, (None, gen_z_dim),
name = “generator_input_z”)
       return real_discrminator_input, generator_inputs_z 
 ```
 
现在要深入构建结构的两个核心组件。我们将从构建生成器部分开始。如图3所示，生成器至少包含2个隐藏层，它将作为一个估算器工作。在这不使用一般的 ReLU
激活函数，而使用Leaky ReLU 激活函数。这将允许梯度值在不受任何约束的情况下通过该层(关于leaky ReLU的下一节将详细介绍)。
