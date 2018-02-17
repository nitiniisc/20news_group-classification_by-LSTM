Folder contains 3 files
1. LSTMclassifier.py
2. LSTMclassifier notebook
3. read_20newsgroup_stemmed.py
first download word2vec pretrained English model from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
 and unzip into data folder
Execute the file  LSTMclassifier that will give accuracy on test data
 ~$ python LSTMclassifier.py      ------using this command
For train data accuracy uncomment the last paragraph code

Requirements:
-python3
-tensorflow
-sklearn
-numpy
-nltk
-gensim
