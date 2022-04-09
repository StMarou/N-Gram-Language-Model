from collections import Counter
import regex
import numpy as np

class LM:
    def __init__(self, n_type):
        """
        n_type: 'bigram' or 'trigram'
        """
        self.n_type = n_type
        if self.n_type not in ['bigram', 'trigram']:
            raise Exception("type must be either 'bigram' or 'trigram'")
        
        self.unique_voc_ = None  # Vocabulary of all the unique words of the corpus
        self.unigram_voc_ = None  # Unigram vocabulary containing words with at least 10 counts
        self.bigram_voc_ = None  # Bigram vocabulary
        self.trigram_voc_ = None  # Trigram vocabulary

        self.previous_words = None  # Number of bigrams where the second element is the keys of the dictionary
        self.following_words = None  # Number of bigrams where the first element is the keys of the dictionary
         
    
    @staticmethod
    def word_cleaner(words):  
        """
        Removes any character that is not a letter or number, excluding the\n
        special tokens.\n
        Input must be a list of words.
        """
        allowed = ['*UNK*', '*start*', '*start1*', '*start2*', '*end*']
        return list(filter(None, 
            [word if word in allowed else regex.sub(r"[^a-zA-Z0-9]", "", word).lower() for word in words]))


    @staticmethod
    def bigram(words):  # create bigrams
        return list(zip(words,words[1:]))
    

    @staticmethod
    def trigram(words):  # create trigrams
        return list(zip(words, words[1:], words[2:]))


    @staticmethod
    def sent_preprocessing(sentences, ngram_padding, padding=True):
        """
        Cleans and pads the sentences according to the type of the model.\n 
        sentences -> must be a list of lists.\n
        ngram_padding -> 2 for bigrams, 3 for trigrams
        padding -> if set to False, then no padding is applied
        """
        clean_sents = [LM.word_cleaner(sent) for sent in sentences]  # clean words
        
        if padding:
            if ngram_padding == 2:  # padding for bigrams (start with *start* and end with *end*)
                padded_sentences = [['*start*'] + sent + ['*end*'] for sent in clean_sents]
            elif ngram_padding == 3:  # trigram padding (each sentence begins with *start1* and *start2* and ends with *end*)
                padded_sentences = [['*start1*'] + ['*start2*'] + sent + ['*end*'] for sent in clean_sents]
            else:
                raise Exception('ngram_padding must be equal to 2 or 3')
        else:
            padded_sentences = clean_sents

        return padded_sentences

    
    @staticmethod
    def unk_replacer(words, voc):
        """
        Replaces the words with a count < 10 with the special token *UNK*.\n
        words -> must be a list of tokens
        voc -> a dictionary with the count of each token
        """
        return [word if voc[word] > 10 and (len(word) > 1 or word in ['i', 'a']) else "*UNK*" for word in words]
    

    def train(self, train_sentences):
        """
        Method to train the chosen language model.\n
        train_sentences must be a list of list, where each list is a different sentence.
        """
        if self.n_type == 'bigram':
            train_sents = LM.sent_preprocessing(train_sentences, 2)  # preprocess the sentences (cleans words and adds padding)
            total_words = [word for sent in train_sents for word in sent]  # flatten list
            
            self.unique_voc_ = Counter(total_words)  # Dictionary with all the counts of the unique tokens

            train_words = LM.unk_replacer(total_words, self.unique_voc_)  # Replace low-count tokens with *UNK*
            self.unigram_voc_ = Counter(train_words)  # Create the unigram vocabulary

            self.bigram_voc_ = Counter(LM.bigram(train_words))  # create a vocabulary of all bigrams
            del self.bigram_voc_[('*end*', '*start*')]

            self.previous_words = Counter(bi[1] for bi in list(self.bigram_voc_.keys()))  # create the voc with the number of previous words
            self.following_words = Counter(bi[0] for bi in list(self.bigram_voc_.keys()))  # create the voc with the number of following words
        else:
            train_sents = LM.sent_preprocessing(train_sentences, 3)  # preprocess sentences for a trigram model
            total_words = [word for sent in train_sents for word in sent]  # flatten list

            self.unique_voc_ = Counter(total_words)

            train_words = LM.unk_replacer(total_words, self.unique_voc_)
            self.unigram_voc_ = Counter(train_words)
            
            self.bigram_voc_ = Counter(LM.bigram(train_words))  # creates bigram voc
            self.trigram_voc_ = Counter(LM.trigram(train_words))  # create trigram voc
            del self.trigram_voc_[('*end*', '*start1*', '*start2*')]
            del self.bigram_voc_[('*end*', '*start1*')]


    def add_a_prob(self, c_ngram, a=1):
        """
        Calculates the probability of a ngram using Add-a smoothing. 
        ngram must be a bigram or a trigram in a tuple.\n
        Default is set to a=1 (laplace smoothing)
        """        
        if a<0 and a>1:
            raise Exception('a must be in [0,1]')

        if self.n_type == 'bigram':
            if len(c_ngram) == 2:
                prob = (self.bigram_voc_[c_ngram] + a) / (self.unigram_voc_[c_ngram[0]] + a * len(self.unigram_voc_))  # add-a probability
            else:
                raise Exception('input is not a bigram')
        else:
            if len(c_ngram) == 3:  # same as above, but for trigrams
                prob = (self.trigram_voc_[c_ngram] + a) / (self.bigram_voc_[c_ngram[:2]] + a * len(self.unigram_voc_))  # |V| unigram bigram or trigram?
            else:
                raise Exception('input is not a trigram')
        return prob


    def kn_prob(self, ngram):
        """
        Calculate the probability of a bigram using Interpolated K-N smoothing.\n
        Available only for bigrams
        """
        if len(ngram) != 2:
            raise Exception('Interpolated K-N Smoothing is only available for bigrams')

        if self.bigram_voc_[ngram] == 1:  # If the count is equal to 1, then steal 0.5
            d = 0.5
        else:  # else steal 0.75
            d = 0.75

        highest_term = max((self.bigram_voc_[ngram] - d), 0) / self.unigram_voc_[ngram[0]]  # first term of the interpolated kn-smoothing formula
        
        following_words = self.following_words[ngram[0]]  # number of words that follow the first element of the bigram

        lamda = (d / self.unigram_voc_[ngram[0]]) * following_words  # interpolation weight

        previous_words = self.previous_words[ngram[1]]  # number of words preceding  the second element of the bigram
        p_cont = previous_words / len(self.bigram_voc_.keys())  # Pcontinuation

        p_kn = highest_term + lamda*p_cont
        return p_kn


    def estimate_sent_prob(self, test_sents, padding=True, a=1, smoothing='add_a', ngram_count=False):
        """
        Estimates the total probability of a sentence.\n
        test_sents must be a list of lists, where each list is a sentence.\n
        padding -> If sentences are already padded set to false\n
        smoothing -> available smoothers: add-a, kn (for Kneser-Ney)\n
        ngram_count -> if set to True, it will also return the total n-gram counts
        """
        if ngram_count not in [True, False]:
            raise Exception('unk can be either se to True or False')

        if a<0 and a>1:
            raise Exception('a must be in [0,1]')

        if smoothing not in ['add_a', 'kn']:
            if self.n_type == 'trigram':
                if smoothing == 'kn':
                    raise Exception('K-N smoothing is only available for bigrams')
            raise Exception('Available smoothers: [add_a, kn]')
        
        sent_probs = []
        total_prob = 0
        ngrams_count = []

        if self.n_type == 'bigram':
            if not padding:
                test_sents = LM.sent_preprocessing(test_sents, 2, padding=False)  # do not pad sentences
            else:
                test_sents = LM.sent_preprocessing(test_sents, 2, padding=True)  # preprocess sentences in order to estimate their probabilities
             
            for sent in test_sents:
                ngrams_count.append(len(LM.bigram(sent)))  # number of bigrams in sentence(useful for cross-entropy calculation)

                for bi in LM.bigram(sent):
                    if smoothing == 'add_a':
                        total_prob += np.log2(self.add_a_prob(bi, a)) # on every iteration, it adds the log probability of each bigram 
                    else:
                        total_prob += np.log2(self.kn_prob(bi))  # use kn

                sent_probs.append(total_prob)  # if multiple sentences are given as an input, it creates a list with the log prob of each sentence
                total_prob = 0  # resets total prob to 0, in order to calculate the probability of the next sentence
        else:
            test_sents = LM.sent_preprocessing(test_sents, 3)

            for sent in test_sents:
                ngrams_count.append(len(LM.trigram(sent)))

                for tri in LM.trigram(sent):
                    total_prob += np.log2(self.add_a_prob(tri, a)) 
                    
                sent_probs.append(total_prob)
                total_prob = 0

        if ngram_count is False:
            return sent_probs
        else:
            return sent_probs, ngrams_count


    def entr_perp(self, test_sents, a=1, smoothing='add_a'):
        """
        Calculates the Cross-entropy and Perplexity of a list of sentences.\n
        test_sents -> must be a list of lists, where each list is a sentence.\n
        smoothing -> 'add_a' or 'kn'
        if smoothing is set to 'add-a', the value of a can be chosen through the parameter a

        """
        test = [LM.unk_replacer(sent, self.unique_voc_) for sent in test_sents]  # replace low-count words with *UNK*
        probs, ngrams_count = self.estimate_sent_prob(test, smoothing=smoothing, a=a, ngram_count=True)  # get the probability of the sentence and its ngram counts
        
        hc = -sum(probs) / sum(ngrams_count)  # calculate cross-entropy
        perp = 2**hc  # calculate perplexity

        return hc, perp