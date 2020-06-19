# How To Make A Text Summarizer

## Overview

- [video](https://youtu.be/ogrJaOIuBx4)
- [encoder-decoder](https://www.tensorflow.org/tutorials/seq2seq)

## Usage

### Data

The video example is made from the text at the start of the article, which I call description (or `desc`),
and the text of the original headline (or `head`). The texts should be already tokenized and the tokens separated by spaces.
[This](http://research.signalmedia.co/newsir16/signal-dataset.html) is a good example dataset. You can use the 'content' as the 'desc' and the 'title' as the 'head'. 

Once you have the data ready save it in a python pickle file as a tuple:
`(heads, descs, keywords)` were `heads` is a list of all the head strings,
`descs` is a list of all the article strings in the same order and length as `heads`.
I ignore the `keywrods` information so you can place `None`.

[Here](http://opendata.stackexchange.com/questions/4981/dataset-of-major-newspapers-content) is a link on how to get similar datasets

### build a vocabulary of words

The [vocabulary-embedding](./vocabulary-embedding.ipynb) notebook describes how a dictionary is built for the tokens and how an initial embedding matrix is built from [GloVe](http://nlp.stanford.edu/projects/glove/).

### train a model

The train.ipynb notebook describes how a model is trained.

`python train.py`

### use model to generate new headlines

The predict.ipynb notebook generate headlines by the trained model and shows the attention weights used to pick words from the description. The text generation includes a feature which was not described in the original paper, it allows for words that are outside the training vocabulary to be copied from the description to the generated headline.

`python predict.py`

### examples of headlines generated

![cherry picking of generated headlines](./cherry_picking.png)
![cherry picking of generated headlines](./cherry_picking1.png)

### examples of attention weights

![attention weights](./attention_weights.png)

## Contributors

- [udibr](https://github.com/udibr)
- [Siraj Raval](https://github.com/llSourcell)
- [B. Bueno](https://github.com/bbueno25)
