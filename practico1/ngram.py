# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import log


class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        for sent in sents:
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

    def prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        return float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]
 
    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.
 
        tokens -- the n-gram or (n-1)-gram tuple.
        """
        
        return self.counts[(tokens)]

 
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]

        count_token = float(self.counts[tuple(tokens)])
        
        if len(prev_tokens) > 0:
            c = 0

        while len(prev_tokens):
            c += float(self.counts[tuple(prev_tokens)])
            prev_tokens = prev_tokens[0:(len(prev_tokens)-1)]

        return count_token/c
                    
 
    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        c = 1

        for i in range(len(sent)-1):
            c *= float(self.counts[tuple(sent)]) / self.counts[tuple(sent[0:len(sent)-1])]
            sent = sen[0:len(sent)-1]

        return c

 
    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """

        c = 1

        for i in range(len(sent)-1):
            c *= log(float(self.counts[tuple(sent)]) / self.counts[tuple(sent[0:len(sent)-1])], 2)
            sent = sen[0:len(sent)-1]

        return c