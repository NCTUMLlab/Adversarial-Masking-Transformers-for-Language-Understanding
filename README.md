# Adversarial-Masking-TransformersforLanguage-Understanding

  Deep  learning  has been achievinggreat  success  incomputer  vision,  natural language processingand many others.Basically, learning representation based on deep neural networks can handle high-dimensional data with complicated mapping between input samplesand output targets and generally perform well for different classification and regression tasks.The sequential machineshavebeen evolving fromlong short-term memory  (LSTM) to transformers andbidirectionalencoder representations  from transformers(BERT). The rapid development of different modelshas enabled deep learning  to buildmany  practicalmodels  fornatural  language  tasks.Basically, transformer is the attention-based encoder-decoder network which has achievedstate-of-art  performance  in machine  translation, text  summarization  and  other  natural language  tasks.BERTcan  be  realizedforlanguage  modelwhich  estimatesthe probability distribution of next wordgiven historywords. Main architectureof BERTis based on the transformer encoder. BERT’straining is divided into two steps: pre-training and fine-tuning.In pre-training stage, the model is trained in an unsupervised learning  manner.  In fine-tuning  stage, labeled  data areused  to  fine-tune modelparameters indifferent tasks,. Since pre-training requires a lot of training samplesand resources, most people use the pre-trainedmodels for fine-tuning. In the pre-training phase, BERT trains with two tricks: masked language model(MLM)and next sentence prediction (NSP).To train MSM, 15% of the words in the sentencesarerandomly replaced with the special word, also known as mask word. BERT is trainedto predict what the masked word is. The NSP trickis to trainBERT to preservethe capability of understandingthe relationship between sentences. Such amodel is trained to determine whether the following sentence is the next sentence of previous sentence. In fine-tuning stage, we can directly initialize model with the pre-trained parameters, and then fine-tune the model with labeled data. Since the number ofparameters required for fine-tuningis limited, the computation costis much less than that in pre-training stage.This study deals with natural language understanding where a deep machine is learned to comprehend contextual meanings of text streamsfrom human languages. A specialized transformer, called the adversarial masking transformer, is proposed. An additional masking module is incorporated to build a hybrid transformer. Masking model aims to mask important information in a sentence. The masked sentence is used as the input to transformer where the encoder and decoder are employed to predict the masked text. Notably, an adversarial learning objective is optimized to guidethe encoder of hybridtransformer to learn the information from the other unmasked texts.Accordingly, the capability of language understanding is improved by attending the important texts, compensating the  missing  information and increasingthe model  robustness. Experiments  on  machine  translation  show  the  merit  of the  proposed  method  for language understandingand sequence-to-sequence learnin

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
