# Adversarial-Masking-Transformers-for-Language-Understanding

  The Masked Language Model(MLM) uses a random method to mask and replace 15$\%$ of the words in the text. Then, learn to predict mask words through the model, let the model learn two-way understanding, and make predictions through context. But because the words are mask randomly,It may mask some words that can be predicted without understanding the context, but this is what we do not want. We hope that each prediction can be completed by the model by understanding the context.Therefore, an effective way is needed to control the words that need to be mask in the text, and the mask words are helpful for the model to learn and understand the context. Therefore, we proposed Adversarial Masking Model.
  
  The structure is shown as the figures. Unlike masked language model, our masking method uses learning instead of randomly masking the input.  When using random masking, some of the more common words (Ex: yes,is,this,...) will also be masked, and because such words appear frequently, the probability of being masked is higher. This is the least we like to see. We don't need to let the model repeatedly learn to guess these common words. We hope that the model is more inclined to guess less common words. Therefore, we propose a dynamic masking module that uses a learning method to select some rare or important words in the input for masking, thereby increasing the probability of the model to guess uncommon words instead of common words.

![image](https://github.com/NCTUMLlab/Adversarial-Masking-TransformersforLanguage-Understanding/blob/main/amt_tm.png)

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
