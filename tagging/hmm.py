from math import log2
from collections import defaultdict


class HMM:

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self.n = n
        self.tags = tagset
        self.trans = trans
        self.out = out

    def tagset(self):
        """Returns the set of tags.
        """
        return self.tags

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.
        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        trans = self.trans
        if (prev_tags in trans) and (tag in trans[prev_tags]):
            return trans[prev_tags][tag]

        return 0.0

    def out_prob(self, word, tag):
        """Probability of a word given a tag.
        word -- the word.
        tag -- the tag.
        """
        if tag in self.out and word in self.out[tag]:
            return float(self.out[tag][word])
        return 0.0

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.
        y -- tagging.
        """
        n = self.n
        tag_prob = 1.0
        y = y + ['</s>']
        prev_y = ('<s>',)*(n-1)

        for tagging in y:
            tag_prob *= float(self.trans_prob(tagging, prev_y))
            if tag_prob == 0.0:
                break
            prev_y += (tagging,)
            prev_y = prev_y[1:]
        return tag_prob

    def prob(self, x, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.
        x -- sentence.
        y -- tagging.
        """
        tag_prob = 1.0
        for tag, word in zip(y, x):
            tag_prob *= self.out_prob(word, tag)
            if tag_prob == 0.0:
                break
        return tag_prob * self.tag_prob(y)

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.
        y -- tagging.
        """
        n = self.n
        y_prob = 0.0
        y = y + ['</s>']
        prev_y = ('<s>',)*(n-1)

        for tagging in y:
            tagging_prob = float(self.trans_prob(tagging, prev_y))
            if tagging_prob == 0.0:
                return float('-inf')
            y_prob += log2(tagging_prob)
            prev_y += (tagging,)
            prev_y = prev_y[1:]
        return y_prob

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.
        x -- sentence.
        y -- tagging.
        """
        log_prob = 0.0

        for tag, word in zip(y, x):
            prob = self.out_prob(word, tag)
            if prob == 0.0:
                break
            log_prob += log2(prob)
        return log_prob + log2(self.tag_prob(y))

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.
        sent -- the sentence.
        """
        viterbitagger = ViterbiTagger(self)
        return viterbitagger.tag(sent)


class ViterbiTagger():

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self.hmm = hmm
        self._pi = {}

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.
        sent -- the sentence.
        """
        n = self.hmm.n
        self._pi = defaultdict(dict)
        prev_tag = ('<s>',) * (n-1)
        self._pi[0][prev_tag] = (log2(1.0), [])
        len_sent = len(sent)
        tagging = []

        for i in range(1, len_sent + 1):
            for tag in self.hmm.tagset():
                out_p = self.hmm.out_prob(sent[i-1], tag)
                if out_p > 0.0:
                    for prev_tags, (log_prob, tags) in self._pi[i-1].items():
                        trans_p = self.hmm.trans_prob(tag, prev_tags)
                        if trans_p > 0.0:
                            tagging = tags + [tag]
                            prob_temp = log_prob + log2(out_p) + log2(trans_p)
                            prev_tag_temp = (prev_tags + (tag,))[1:]
                            if prob_temp > self._pi[i].get(prev_tag_temp,
                                                           (float('-inf'),)
                                                           )[0]:
                                self._pi[i][prev_tag_temp] = (prob_temp,
                                                              tagging)

        tag_final = []
        prob = float('-inf')
        for prev_tags, (log_prob, tags) in self._pi[len_sent].items():
            trans = self.hmm.trans_prob('</s>', prev_tags)
            lp = log_prob + log2(trans)
            if lp > prob:
                tag_final = tags
                prob = lp
        return tag_final


class MLHMM(HMM):

    def __init__(self, n, tagged_sents, addone=True):
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        self.n = n
        self.addone = addone
        self.tagged_sents = tagged_sents
        self.count = count = defaultdict(int)
        self.out = out = defaultdict(int)
        self.vocabulary = vocabulary = set()
        self.tags = tags = set()

        for sent in tagged_sents:
            if sent != []:
                w, t = zip(*sent)
                t = t + ('</s>',)
                w = w + ('</s>',)
                h = 0
                for j in range(n+1):
                    for i in range(len(t)-j+h):
                        if j == 0:
                            if i != len(w)-1:
                                vocabulary.add(w[i])
                            tags.add(t[i])
                            out[(t[i], w[i])] += 1
                        count[tuple(t[i:i+j])] += 1
                    t = ('<s>',)*(n-1) + t
                    h = 1

        self.len_tags = len(tags)
        self.len_voc = len(vocabulary)

    def tcount(self, tokens):
        """Count for an k-gram for k <= n.
        tokens -- the k-gram tuple.
        """
        return self.count[tokens]

    def unknown(self, w):
        """Check if a word is unknown for the model.
        w -- the word.
        """
        if w in self.vocabulary:
            return False
        return True

    def out_prob(self, word, tag):
        """Probability of a word given a tag.
        word -- the word.
        tag -- the tag.
        """
        if self.count[(tag, )] == 0:
            result = 0
        elif self.unknown(word):
            result = (1/self.len_voc)
        else:
            result = self.out[(tag, word)] / self.count[(tag,)]
        return result

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.
        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        if self.addone:
            result = ((self.count[prev_tags + (tag, )] + 1) /
                      (self.count[prev_tags] + self.len_tags))
        elif self.count[prev_tags] == 0:
            result = 0.0
        else:
            result = (self.count[prev_tags + (tag,)] /
                      self.count[prev_tags])

        return result
