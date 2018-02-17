from sklearn.datasets import fetch_20newsgroups
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

class ReadNewsGroup(object):
    """
        Reads the 20 newsgroup dataset document by document with stemming, 
        removes stopwords and removes words of length <= 1
    """
    
    def __init__(self, data_dir, categories=None, rare_threshold=0, seed=None):
        """
            __init__(ReadNewsGroup, str, list, int, float) -> None
            data_dir: Path where raw dataset should be saved
            categories: List of str, categories to include
            seed: Random seed to use
            rare_threshold: Words which occur <= rare_threshold times will be 
                            replaced by __rare__
        """
        
        # Some useful variables
        self.rare_threshold = rare_threshold
        self.num_categories = 0
        self.categories = []
        self.stemmer = SnowballStemmer('english')
        self.stop_words = [self.stemmer.stem(w) for w in \
                                                set(stopwords.words('english'))]
        
        # Initialize the state of random number generator
        temp = np.random.get_state()
        if seed != None:
            np.random.seed(seed)
        self.state = np.random.get_state()
        np.random.set_state(temp)
        
        # Get the train and test set
        self.train_set = fetch_20newsgroups(data_dir, subset='train', \
                 categories=categories, remove=('headers', 'footers', 'quotes'))
        self.test_set = fetch_20newsgroups(data_dir, subset='test', \
                 categories=categories, remove=('headers', 'footers', 'quotes'))
        
        # Read the documents
        self.read_data()
                
                
    
    def read_data(self):
        """
            read_data(ReadNewsGroup) -> (list, list, list, list)
            Reads the training and test dataset and delimits sentences
            
            Returns:
                train_data: List of training sentences as list of idx
                test_data: List of test sentences as list of idx
                train_labels: List of sentence labels for training set
                test_labels: List of sentence labels for test set
        """
        # Read the categories
        self.categories = self.train_set.target_names
        self.num_categories = len(self.categories)
        
        # Create the vocabulary
        self.build_vocab()
        
        # Rarify
        self.rarify()
        
        chars = 'abcdefghijklmnopqrstuvwxyz '
        
        # Create documents for train set
        self.train_data = []
        self.train_labels = []
        i = 0
        for document in self.train_set.data:
            email = ''
            document = document.lower()
            for char in document:
                if char in chars:
                    email += char
                else:
                    email += ' '
            
            
            words = email.split(' ')
            doc = []
            for word in words:
                #word = self.stemmer.stem(word)
                if len(word) > 1 and word not in self.stop_words:
                    if word in self.vocab:
                        doc.append(word)
             
            
            if len(doc) > 0:
                self.train_data.append(doc)
                self.train_labels.append(self.train_set.target[i])
            i += 1
        
        
        # Create documents for test set
        self.test_data = []
        self.test_labels = []
        i = 0
        for document in self.test_set.data:
            email = ''
            document = document.lower()
            for char in document:
                if char in chars:
                    email += char
                else:
                    email += ' '
            
            
            words = email.split(' ')
            doc = []
            for word in words:
                #word = self.stemmer.stem(word)
                if len(word) > 1 and word not in self.stop_words:
                    if word in self.vocab:
                        doc.append(word)
            
            if len(doc) > 0:
                self.test_data.append(doc)
                self.test_labels.append(self.test_set.target[i])
            i += 1
        
        
        
    
    def build_vocab(self):
        """
            build_vocab(ReadNewsGroup) -> None
            Generates the vocabulary from word to index and from index to word
            and computes word count
        """
        chars = 'abcdefghijklmnopqrstuvwxyz '
        
        # Initialize the structures
        self.vocab = dict()
        self.vocab_idx = dict()
        token = 0
        
        # Read the training set
        for document in self.train_set.data:
            email = ''
            document = document.lower()
            for char in document:
                if char in chars:
                    email += char
                else:
                    email += ' '
            
            words = email.split(' ')
            for word in words:
                #word = self.stemmer.stem(word)
                if len(word) > 1 and word not in self.stop_words:
                    if word not in self.vocab:
                        self.vocab[word] = [token, 1]
                        self.vocab_idx[token] = [word, 1]
                        token += 1
                    else:
                        idx = self.vocab[word][0]
                        self.vocab[word][1] += 1
                        self.vocab_idx[idx][1] += 1
            
        # Read the test set
        for document in self.test_set.data:
            email = ''
            document = document.lower()
            for char in document:
                if char in chars:
                    email += char
                else:
                    email += ' '
            
            words = email.split(' ')
            for word in words:
                #word = self.stemmer.stem(word)
                if len(word) > 1 and word not in self.stop_words:
                    if word not in self.vocab:
                        self.vocab[word] = [token, 1]
                        self.vocab_idx[token] = [word, 1]
                        token += 1
                    else:
                        idx = self.vocab[word][0]
                        self.vocab[word][1] += 1
                        self.vocab_idx[idx][1] += 1
                        
    
    
    def rarify(self):
        """
            rarify(ReadNewsGroup) -> None
            Removes the rare words from the vocabulary
        """
        if self.rare_threshold <= 0:
            return
            
        temp_vocab = dict()
        temp_vocab_idx = dict()
        
        token = 0
        temp_vocab['__rare__'] = [token, 0]
        temp_vocab_idx[token] = ['__rare__', 0]
        token += 1
        
        for word in self.vocab:
            if self.vocab[word][1] <= self.rare_threshold:
                temp_vocab['__rare__'][1] += 1
                temp_vocab_idx[0][1] += 1
            else:
                temp_vocab[word] = (token, self.vocab[word][1])
                temp_vocab_idx[token] = (word, self.vocab[word][1])
                token += 1
        
        self.vocab = temp_vocab
        self.vocab_idx = temp_vocab_idx
                        
                        
    def get_data(self, batch_size, set_type='train'):
        """
            get_data(ReadNewsGroup, int, str) -> (list of list, list)
            batch_size: Number of elements in the batch
            set_type: 'train', 'test'
            
            Prepares a batch of data
            
            Returns:
                emails: List of list containing emails
                labels: Category labels
        """
        emails = []
        labels = []
        
        # Set up random sampler
        temp = np.random.get_state()
        np.random.set_state(self.state)
        
        if set_type == 'train':
            idx = np.random.choice(len(self.train_data), size=(batch_size,), \
                                                                replace=False)
            for i in range(batch_size):
                emails.append(self.train_data[idx[i]])
                labels.append(self.train_labels[idx[i]])
        
        elif set_type == 'test':
            idx = np.random.choice(len(self.test_data), size=(batch_size,), \
                                                                replace=False)
            for i in range(batch_size):
                emails.append(self.test_data[idx[i]])
                labels.append(self.test_labels[idx[i]])
        
        
        # Reset random sampler
        self.state = np.random.get_state()
        np.random.set_state(temp)
        
        return (emails, labels)
        
        
        
    def word_to_idx(self, word):
        return self.vocab[word][0]
        
    def idx_to_word(self, idx):
        return self.vocab_idx[idx][0]
    
    def word_count(self, word=None, idx=None):
        if word is not None:
            return self.vocab[word][1]
        elif idx is not None:
            return self.vocab_idx[idx][1]
            
    def idx_to_cat(self, idx):
        return self.categories[idx]
        
        
        
if __name__ == '__main__':
    arxiv = ReadNewsGroup('./data/', rare_threshold=5, seed=10)
    emails, labels = arxiv.get_data(20, set_type='train')
    #print(emails)
    #print(labels)
    k = 1
    
    for k in range(20):
        temp = emails[k]
        n = len(temp)
        for i in range(n):
            print(arxiv.idx_to_word(temp[i]) + ' ', end='')
            
        print('\n' + arxiv.idx_to_cat(labels[k]))
    print(len(arxiv.vocab))
