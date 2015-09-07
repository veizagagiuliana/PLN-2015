# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import random
from math import log2


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
            s = sent + ['</s>']            
            for i in range(n-1):
                s = ['<s>'] + s
            for i in range(len(s) - n + 1):
                ngram = tuple(s[i: i + n])
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

        return self.counts[tokens]

 
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
        if self.counts[tuple(tokens[:-1])] == 0:
            return float(self.counts[tuple(tokens)]) / float('inf')
        return float(self.counts[tuple(tokens)]) / float(self.counts[tuple(prev_tokens)])
  
    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        c = 1
        sent = sent + ['</s>']
        s = sent

        for x in range(self.n-1):
            s = ['<s>'] + s

        for i in range(len(sent)):
            if self.n == 1:
                c *= float(self.cond_prob(sent[i]))
            else:
                c *= float(self.cond_prob(sent[i], s[i:self.n+i-1]))
        return c

 
    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """
        x = self.sent_prob(sent)
        if x == 0.0:
            return float('-inf')
        return log2(x)



class NGramGenerator:
 
    def __init__(self, model):
        """
        model -- n-gram model.
        """

        assert model.n > 0
        self.n = model.n
        self.probs = probs = defaultdict(dict)
        self.sorted_probs = sorted_probs = defaultdict(list)

        for token in model.counts:
            if len(token) == self.n:
                name = token[:-1]
                dic = token[-1:]
                probs[name][dic[0]] = model.cond_prob(dic[0],list(name))
                sorted_probs[name].append(tuple((dic[0], model.cond_prob(dic[0],
                                                 list(name)))))
                sorted_probs[name].sort()


    def generate_sent(self):
        """Randomly generate a sentence."""

        n = self.n
        s = []
        sent = []
        for x in range(n-1):
            s += ['<s>']
        while True:
            wn = self.generate_token(tuple(s))
            if wn != '</s>':
                sent += [wn]
                if tuple(sent)[-1:][0] is '.':
                    break
                if n!=1:
                    s = s[1:]
                    s += [wn]
        return sent

 
    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.
 
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        if not prev_tokens:
            prev_tokens = ()
        assert (len(prev_tokens) == n - 1)
        words = self.sorted_probs[tuple(prev_tokens)]
        large = len(words)
        r = random.random()
        k = 0

        for i in range(large):
            # print(words[i][0])
            if (k < r) and (r <= k + words[i][1]):
                return words[i][0]
            else:
                k += words[i][1]

        return