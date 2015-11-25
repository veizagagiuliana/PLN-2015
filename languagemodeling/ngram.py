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
            sent = ['<s>']*(n-1) + sent + ['</s>']
            for i in range(len(sent) - (n - 1)):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

    def prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        return float(self.counts[tuple(tokens)]) / \
            self.counts[tuple(prev_tokens)]

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
        return float(self.counts[tuple(tokens)]) / \
            float(self.counts[tuple(prev_tokens)])

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
        sent -- the sentence as a list of tokens.
        """
        sent_prob = 1.0
        sent = sent + ['</s>']
        prev_tokens = ['<s>']*(self.n-1)

        for token in sent:
            sent_prob *= float(self.cond_prob(token, prev_tokens))
            if sent_prob == 0.0:
                return sent_prob
            prev_tokens = (prev_tokens + [token])[1:]
        return sent_prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        sent_prob = 0.0
        sent = sent + ['</s>']
        prev_tokens = ['<s>'] * (n - 1)

        for token in sent:
            token_prob = float(self.cond_prob(token, prev_tokens))
            if token_prob == 0.0:
                return float('-inf')
            sent_prob += log2(token_prob)
            prev_tokens = (prev_tokens + [token])[1:]
        return sent_prob

    def log_probability(self, sents):
        sum_log = 0
        for sent in sents:
            sum_log += self.sent_log_prob(sent)
        return sum_log

    def cross_entropy(self, sents):
        M = 0
        for sent in sents:
            M += len(sent)
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
                probs[name][dic] = model.cond_prob(dic, list(name))
                sorted_probs[name].append((dic, model.cond_prob(dic,
                                                                list(name))))
                sorted_probs[name].sort()
                sorted_probs[name].sort(key=lambda tup: tup[1], reverse=True)
        for prev, l in sorted_probs.items():
            assert abs(sum(x[1] for x in l) - 1.0) < \
                1e-10, (prev, sum(x[1] for x in l))

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
            if n != 1:
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
        self.n = n
        self.v = self.V()

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        count = self.count
        v = self.v
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]

        return (float(count(tuple(tokens))) + 1.0) / \
            (float(count(tuple(prev_tokens)) + v))

    def V(self):
        """Size of the vocabulary.
        """
        n = self.n
        voc = []
        for grams, c in self.counts.items():
            if len(grams) == n:
                for g in grams:
                    voc += [g]
        voc = list(set(voc))
        if '<s>' in voc:
            voc.remove('<s>')
        v = len(voc)
        return v


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        NGram.__init__(self, n, sents)
        len_sents = len(sents)

        self.addone = addone
        self.v = self.V()

        if gamma is not None:
            self.gamma = gamma
        else:
            held_out = sents[int(0.9 * len_sents):]
            sents = sents[:int(0.9 * len_sents)]
            self.build_gamma(held_out)

        self.counts = self.build_count(sents)

    def build_count(self, sents):
        n = self.n
        counts = defaultdict(int)
        for sent in sents:
            sent = ['<s>'] * (n-1) + sent + ['</s>']
            len_sent = len(sent)
            for i in range(1, n+1):
                for j in range(len_sent - (n - 1)):
                    ngram = tuple(sent[len_sent-i-j:len_sent-j])
                    counts[ngram] += 1
                    if i == 1:
                        counts[tuple()] += 1
                if i != 1:
                    counts[tuple(['<s>']*i)] += 1
        return counts

    def build_gamma(self, held_out):
        self.gamma = 1
        gamma_ok = self.gamma
        old_perp = self.perplexity(held_out)
        for gamma in range(20, 1000, 20):
            self.gamma = gamma
            new_perp = self.perplexity(held_out)
            if new_perp < old_perp:
                old_perp = new_perp
                gamma_ok = self.gamma
        self.gamma = gamma_ok

    def cond_prob_ML(self, token, prev_tokens=None):
        addone = self.addone
        if not prev_tokens:
            prev_tokens = []

        tokens = prev_tokens + [token]
        if not addone or len(prev_tokens) != 0:
            if float(self.count(tuple(prev_tokens))) != 0.0:
                result = (float(self.count(tuple(tokens))) /
                        float(self.count(tuple(prev_tokens))))
            else:
                result = 0.0
        else:
            result = (float(self.count(tuple(tokens)) + 1) /
                    (float(self.count(tuple(prev_tokens))) + float(self.v)))
        return result

    def lamdas(self, tokens):
        n = self.n
        lamdas = []
        for i in range(n - 1):
            num = (1 - sum(lamdas)) * self.count(tuple(tokens[i:]))
            den = self.count(tuple(tokens[i:])) + self.gamma
            lamda = num / den
            lamdas.append(lamda)
        lamda = float(1 - sum(lamdas))
        lamdas.append(lamda)
        return lamdas

    def cond_prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        lamdas = self.lamdas(prev_tokens)
        prob = 0.0
        for i in range(n):
            if lamdas[i] != 0:
                prob += lamdas[i] * self.cond_prob_ML(token, prev_tokens[i:])
        return prob

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


class BackOffNGram(NGram):

    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        NGram.__init__(self, n, sents)
        self.addone = addone

        if beta is None:
            held_out = sents[int(len(sents)*0.9):]
            sents = sents[:int(len(sents)*0.9)]

        self.counts = counts = self.build_count(sents)
        self.a = defaultdict(set)
        self.dict_denom = dict()
        self.dict_alpha = dict()
        self.v = 0

        if addone:
            self.V()

        for token, count in counts.items():
            # usado en self.A()
            if len(token) > 1:
                self.a[tuple(token[:-1])].add(token[-1])
                if "<s>" in self.a[tuple(token[:-1])]:
                    self.a[tuple(token[:-1])].remove("<s>")

        if beta is None:
            self.beta = self.beta(held_out)
        else:
            self.beta = beta
            self.build_alphas()
            self.build_denom()

    def build_alphas(self):
        beta = self.beta
        dict_alpha = dict()
        for tokens in self.a.keys():
            count = self.count(tokens)
            list_postokens = list(self.A(tokens))
            num_postokens = len(list_postokens)
            if count != 0:
                dict_alpha[tokens] = (beta * num_postokens)/count
        self.dict_alpha = dict_alpha

    def build_denom(self):
        dict_denom = dict()
        for tokens in self.a.keys():
            list_postokens = list(self.A(tokens))
            prev_tokens = list(tokens)
            p = 0.0
            for token in list_postokens:
                p += self.cond_prob(token, prev_tokens[1:])
            dict_denom[tokens] = 1 - p
        self.dict_denom = dict_denom

    def build_count(self, sents):
        n = self.n
        counts = defaultdict(int)
        for sent in sents:
            sent = ['<s>']*(n-1) + sent + ['</s>']
            len_sent = len(sent)
            for i in range(1, n+1):
                for j in range(len_sent - (n - 1)):
                    ngram = tuple(sent[len_sent-i-j:len_sent-j])
                    counts[ngram] += 1
                    if i == 1:
                        counts[tuple()] += 1
                if i != 1:
                    counts[tuple(['<s>']*(i-1))] += 1
        return counts

    def beta(self, sents):
        self.beta = 0.1
        beta_ok = self.beta
        old_perp = self.perplexity(sents)
        for beta in range(1, 11):
            self.build_alphas()
            self.build_denom()
            self.beta = beta * 0.1
            new_perp = self.perplexity(sents)
            if new_perp < old_perp:
                old_perp = new_perp
                beta_ok = self.beta
        return beta_ok

    def count_prime(self, tokens):
        return self.count(tokens) - self.beta

    def cond_prob_ML(self, token, prev_tokens=None):
        addone = self.addone
        if not prev_tokens:
            prev_tokens = []

        tokens = prev_tokens + [token]
        if not addone or len(prev_tokens) != 0:
            if float(self.count(tuple(prev_tokens))) != 0.0:
                result = (float(self.count(tuple(tokens))) /
                        float(self.count(tuple(prev_tokens))))
            else:
                result = 0.0
        else:
            result = (float(self.count(tuple(tokens)) + 1) /
                    (float(self.count(tuple(prev_tokens))) + float(self.v)))
        return result

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if not prev_tokens:
            prev_tokens = []

        tokens = prev_tokens + [token]
        if len(prev_tokens) == 0:
            prob = self.cond_prob_ML(token, prev_tokens)
        else:
            A = self.A(tuple(prev_tokens))
            if token in A:
                prob = self.count_prime(tuple(tokens)) / \
                       self.count(tuple(prev_tokens))
            else:
                if len(prev_tokens) < 1:
                    cond_prob = self.cond_prob_ML(token, prev_tokens[1:])                    
                else:
                    cond_prob = self.cond_prob(token, prev_tokens[1:])
                alpha = self.alpha(tuple(prev_tokens))
                denom = self.denom(tuple(prev_tokens))
                prob = alpha * (cond_prob / denom)
        return prob

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """
        return self.a[tokens]

    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """

        if tokens in self.dict_alpha:
            return self.dict_alpha[tokens]
        return 1

    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """
        if tokens in self.dict_denom:
            return self.dict_denom[tokens]
        return 1

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
        self.v = len(v)
