# Adversarial-Masking-TransformersforLanguage-Understanding

  Sequential learning is particular research topic in deep learning. It focuses on sequential data. There is no limitation on what type of the element contained in a sequence. It can be word for a sentence, a value for a signal, or even a image for a videostream. The goal of sequential learning is mainly to extract the hidden features from the sequential data, and this also implies that it tries to understand the meaning of the sequence. As long as the computer can understand a sequence, it can be applied to a wide variety of applications. But like we mentioned above, the type of the element in a sequence is various. In this work, we focus more on the natural language instead of computer vision. For computer vision(CV), the sequential data usually is videostream. However, learning the dependency between frames might not be the most crucial part in the related tasks like object tracking, video object segmentation.

  Although learning good dependency can improve the performance, the more important thing is to extract good features for each frame. Even though, we only obtain a single frame of the video, we can know what and how many ojects are and can guess what is happening. In contrast to CV, NLP needs to learn good dependency not just the features in a single element in the sequence. Single element like a word or a value in NLP doesn't provide with information. It requires to know the surrounding elements, then we can understand some meaning of the sequence. For example, given "good", a single token, the following two sentences "It is not good at all." and "It is so good.", they have totally opposite meaning. Of course, the meaning of single element is crucial, but the dependency might play a more important role. 
  
  Then, how do we evaluate if a computer know the meaning of a sentence. In fact, nowadays, we don't have an universal neural network that can fully understand the meaning of a sequence. But we can view the meaning of "understanding" in different perspective. For instance, people have the ability to guess the possible elements in an incomplete sequence, answer a question, know if the two sequences are similar etc. These are individual tasks in current NLP research. Though it is possible that we share some modules in different tasks, it is still not an universal network. Despite no universal network for understanding the meaning of a sequence, there is a special network called "Language model" widely used in most of these tasks.

## Environment

The developed environment is listed in below 

OS : Ubuntu 16.04 

CUDA : 11.1

Nvidia Driver : 455.23

Python 3.6.9

Pytorch 1.2.0

## Installation

PyTorch version >= 1.5.0

Python version >= 3.6

For training new models, you'll also need an NVIDIA GPU and NCCL

To install fairseq and develop locally:

```sh
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```

## Training and Evaluation

```sh

./run.sh

```


## Reference

https://github.com/pytorch/fairseq
