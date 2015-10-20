from math import log2, log
from collections import defaultdict
from operator import itemgetter

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
        n = self.n
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
                return tag_prob
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
        n = self.n
        tag_prob = 1.0

        for tag, word in zip(y, x):
            tag_prob *= self.out_prob(word, tag)
            if tag_prob == 0.0:
                return tag_prob
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
        return log2(self.prob(x,y))

 
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
        tagging = []

        for i in range(1, len(sent)+1):
            for prev_tags, (log_prob, tags) in self._pi[i-1].items():
                for tag in self.hmm.tagset():
                    out_p = self.hmm.out_prob(sent[i-1], tag)
                    trans_p = self.hmm.trans_prob(tag, prev_tags)
                    if out_p * trans_p != 0:
                        tagging = tags + [tag]
                        prob_temp = log_prob + log2(out_p * trans_p)
                        prev_tag_temp = prev_tags[1:] + tuple(tag)
                        try:
                            prob_actual = self._pi[i][prev_tag_temp][0]
                        except:
                            prob_actual = float('-inf')

                        if prob_temp > prob_actual:
                            self._pi[i][prev_tag_temp] = (prob_temp, tagging)

        tag_final = []
        prob = 0.0
        for prev_tags, (log_prob, tags) in self._pi[len(sent)].items():
            trans = self.hmm.trans_prob('</s>', prev_tags)
            if trans > prob:
                tag_final = tags
                prob = trans

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
        self.vocabulary = vocabulary = []
        self.tags = tags = []

        tag = []

        for sent in tagged_sents:
            if sent != []:
                w, t = zip(*sent)
            t = t + ('</s>',)
            w = w + ('</s>',)
            h = 0
            for j in range(n+1):
                for i in range(len(t)-j+h):
                    if j == 0:
                        vocabulary.append(w[i])
                        tags.append(t[i])
                        out[(t[i], w[i])] += 1
                    count[tuple(t[i:i+j])] += 1
                t = ('<s>',)*(n-1) + t
                h = 1
        vocabulary = set(vocabulary)
        tags = set(tags)
        #feo, revisar

    def tcount(self, tokens):
        """Count for an k-gram for k <= n.
 
        tokens -- the k-gram tuple.
        """
        return self.count[tokens]

    def unknown(self, w):
        """Check if a word is unknown for the model.
 
        w -- the word.
        """
        # for word in self.vocabulary:
        #     if word == w:
        #         return False
        if w in self.vocabulary:
            return False
        return True

    def out_prob(self, word, tag):
        """Probability of a word given a tag.
 
        word -- the word.
        tag -- the tag.
        """
        if self.unknown(word):
            return (1/len(self.vocabulary))

        if self.tcount((tag ,)) == 0:
            return 0

        return self.out[(tag, word)] / self.tcount((tag,))

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.
 
        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        n = self.n

        if self.addone:
            return ((self.tcount(prev_tags + (tag,)) + 1 )/ 
                    (self.tcount(prev_tags) + len(self.tags)))
        return (self.tcount(prev_tags + (tag,))/ 
                self.tcount(prev_tags))
