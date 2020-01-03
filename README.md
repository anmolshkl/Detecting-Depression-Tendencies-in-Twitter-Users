# Detecting Depression Tendencies in Twitter Users

In this project, we have implemented different models for detecting depression in Twitter Users. More details regarding this project can be found in the PDF report in the repository.

## Files

The Project consists of the following files-

1. BaselineNLP.ipynb: This file consists of our baseline code which is CountVectorizer and TfIdf with Logistic Regression. The usage of CountVectorizer and TfIdf with ngrams is taken from the following website: [CountVectorizer and Tfidf with Ngrams](https://medium.com/@annabiancajones/sentiment-analysis-on-reviews-feature-extraction-and-logistic-regression-43a29635cc81) 
2. DatasetBias.ipynb: This file is used to analyze bias in our dataset.
3. LR_GloveEmbeddings.ipynb: This file contains a logistic regression model trained using Twitter corpus Glove embeddings. 
4. NeuralModels.ipynb: This file contains our experimentation with different neural models such as LSTM + CNN, BiLSTM, BiLSTM with Attention. The architecture used in our code is inspired from [BiLSTM With Attention] and BiLSTM code is built with the help of [Keras documentation](https://keras.io/examples/imdb_bidirectional_lstm/). 
5. RoBERTa.ipynb: This file contains the code for training RoBERTa Model using SimpleTransformers Library which is built on Hugging Face Library. We have used the [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers) library.


## Dataset
We manually annotated around 5k Tweets which included about 2k depressive Tweets. To match the distribution of the source i.e. Twitter, we added Tweets from other different sources and marked them as non-depressive. Unfortunately, we cannot share the dataset as per Twitter's policies. However, we can share the Tweet IDs which can be used to scrape the Tweets from Twitter.

## Notebook
We have uploaded the .ipynb files and all our results of our experiments can be seen directly from the file. The code can be run directly from the notebook (provided you have the dataset).

## Requirements
You will need the following libraries:
1. Python 3.6
2. Tensor flow tf 2.0
3. Keras 2.3
4. PyTorch 1.2
5. TorchVision 0.4.0
6. Nvidia/Apex
7. SimpleTransformers
8. Hugging Face
9. Jupyter Notebook