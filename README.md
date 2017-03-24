# NLP_ps1
CS224n (2017) assignment1 coding problems

Some of this code is adapted from anothers' solutions to the problem set from two or three years ago.  (I'll upload my original version when I have time to annotate it.)  I had hoped that the instant code would be more tractable to parallelizing, but that achievement remains elusive. 

Notes:

The cross-entropy cost function, with the skip-gram model, seems to plateau at around 9.4.  Costs with the cbow model are substantially lower, but that may stem from a coding error.  Note that  to run the cbow model you must replace 'skipgram' with 'cbow' in the invocation of the  word2vec_sgd_wrapper contained in q3_run.py.

None of this may be correct.

For the sentiment problem, the best regularization (determined on dev set) gave test accuracies of 29.5% (skipgram) and 28.9 (cbow) compared to 37.1%  with pretrained word vectors.  


