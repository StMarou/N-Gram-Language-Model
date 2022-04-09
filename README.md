
# N-Gram Language Model from scratch

A Python implementation of an N-Gram Language Model. There are two
available types of n-gram models (specified using the *n_type* 
parameter): a bigram model and a trigram model.

For the bigram model, two smoothers are available: the add-a smoother
(defaulted to a=1 to apply laplace smoothing) and the interpolated
Kneser-Ney smoother. For the trigram model, only the add-a smoother
is available.

In the *example* notebook, using *NLTK*'s 'abc' corpus, the Language
model is used. First, its methods are described and presented through
some examples. Then some applications are presented, such as assigning
probabilities to sentences, calculating the cross-entropy and perplexity
of a corpus and predicting the most probable word continuations for a 
specific token. 

