# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import random
import pdb
from math import log2, log


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
        if self.counts[tuple(prev_tokens)] == 0:
            return float(self.counts[tuple(tokens)]) / float('inf')
        return float(self.counts[tuple(tokens)]) / float(self.counts[tuple(prev_tokens)])

  
    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        sent_prob = 1.0
        sent = sent + ['</s>']
        prev_tokens = ['<s>']*(self.n-1)

        for i in range(len(sent)):
            sent_prob *= float(self.cond_prob(sent[i], prev_tokens))
            prev_tokens.append(sent[i])
            prev_tokens = prev_tokens[1:]
        return sent_prob


    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        sent_prob = 0.0
        sent = sent + ['</s>']
        prev_tokens = ['<s>']*(n-1)

        for token in sent:
            token_prob = float(self.cond_prob(token, prev_tokens))
            if token_prob == 0.0:
                return float('-inf')
            sent_prob += log(token_prob, 2)
            prev_tokens.append(token)
            prev_tokens = prev_tokens[1:]
        return sent_prob


    def log_probability(self, sents):
        sum_log = 0.0
        for sent in sents:
            sum_log += self.sent_log_prob(sent)
        return sum_log


    def cross_entropy(self, sents):
        M = 0
        for sent in sents:
            M += len(sent) + 1
        sum_log = self.log_probability(sents)
        return (-sum_log/M)


    def perplexity(self, sents):
        ce = self.cross_entropy(sents)
        return (2**ce)


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
                dic = token[-1]
                probs[name][dic] = model.cond_prob(dic,list(name))
                sorted_probs[name].append((dic, model.cond_prob(dic,
                                                 list(name))))
                sorted_probs[name].sort()
                sorted_probs[name].sort(key=lambda tup: tup[1], reverse=True)
        for prev, l in sorted_probs.items():
            assert abs(sum(x[1] for x in l) - 1.0) < 1e-10, (prev, sum(x[1] for x in l))

    def generate_sent(self):
        """Randomly generate a sentence."""

        n = self.n
        s = []
        sent = []
        for x in range(n-1):
            s += ['<s>']
        while True:
            wn = self.generate_token(tuple(s))
            if wn == '</s>':
                break
            sent += [wn]
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
        k = 0.0

        for i in range(large):
            if (k < r) and (r <= k + words[i][1]):
                return words[i][0]
            else:
                k += words[i][1]

        assert abs(k - 1.0) < 1e-10
        assert abs(r - 1.0) < 1e-10
        return words[i][0]

class AddOneNGram(NGram):
 
    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        NGram.__init__(self, n, sents)
        self.v = self.V()
        
 
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
        if self.count(tuple(tokens[:-1])) == 0:
            return (float(self.count(tuple(tokens))) + 1.0) \
                            / (float('inf') + self.v)

        result = (float(self.count(tuple(tokens))) + 1.0) \
                        / (float(self.count(tuple(prev_tokens))) + self.v)

        if self.counts[tuple(tokens)] == 0:
            del self.counts[tuple(tokens)]

        return float(result)


    def V(self):
        """Size of the vocabulary.
        """
        v = []
        for w, c in self.counts.items():
            if len(w) == self.n:
                for i in w:
                    v += [i]
        v = list(set(v))
        if '<s>' in v:
            v.remove('<s>')
        return len(v)

class InterpolatedNGram:
 
    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
 
    def count(self, tokens):
        """Count for an k-gram for k <= n.
 
        tokens -- the k-gram tuple.
        """
 
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

class BackOffNGram:
 
    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.
 
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
 
    def count(self, tokens):
        """Count for an k-gram for k <= n.
 
        tokens -- the k-gram tuple.
        """
 
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
 
    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
 
    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
 
    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """