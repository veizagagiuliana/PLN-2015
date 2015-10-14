from math import log2
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
        self.tag_set = tagset
        self.trans = trans
        self.out = out

 
    def tagset(self):
        """Returns the set of tags.
        """
        return self.tag_set
 
    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.
 
        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        n = self.n
        if not prev_tags:
            prev_tags = []
        assert len(prev_tags) == n - 1

        return self.trans[tuple(prev_tags)][tag]

    def out_prob(self, word, tag):
        """Probability of a word given a tag.
 
        word -- the word.
        tag -- the tag.
        """
        if word in self.out[tag]:
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
        prev_y = ['<s>']*(n-1)

        for tagging in y:
            tag_prob *= float(self.trans_prob(tagging, prev_y))
            if tag_prob == 0.0:
                return tag_prob
            prev_y.append(tagging)
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
        return tag_prob
 
    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.
 
        y -- tagging.
        """
        n = self.n
        y_prob = 0.0
        y = y + ['</s>']
        prev_y = ['<s>']*(n-1)

        for tagging in y:
            tagging_prob = float(self.trans_prob(tagging, prev_y))
            if tagging_prob == 0.0:
                return float('-inf')
            y_prob += log2(tagging_prob)
            prev_y.append(tagging)
            prev_y = prev_y[1:]
        return y_prob

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.
 
        x -- sentence.
        y -- tagging.
        """
        log_prob = 0.0
        for word, tag in zip(x, y):
            prob = self.out_prob(word, tag)
            if prob == 0.0:
                return float('-inf')
            log_prob += log2(prob)
        return log_prob

 
    def tag(self, sent):
        """Returns the most probable tagging for a sentence.
 
        sent -- the sentence.
        """
        # tagging = []
        # for word in sent:
        #     tags = []
        #     for tag in self.tagset():
        #         tags.append((tag, self.out_prob(word, tag)))
        #     tagging.append(max(tags, key=itemgetter(1))[0])
        # return tagging

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
        len_tag = 0
        log_prob = 0.0
        tagging = []
        self._pi[len_tag][prev_tag] = (log2(1.0), [])

        for word in sent:
            tags = []
            for tag in self.hmm.tagset():
                tags.append((tag, self.hmm.out_prob(word, tag)))
            tag = [max(tags, key=itemgetter(1))[0]]
            tagging = tagging + [tag[0]]
            len_tag += 1
            prev_tag = prev_tag[1:] + tuple(tag[0])
            log_prob += self.hmm.log_prob([word], tag[0])
            self._pi[len_tag][prev_tag] = (log_prob, tagging)
        return tagging


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
        tag = []

        for sent in tagged_sents:
            sent = sent + [('</s>', '</s>')]
            for i in range(0, n+1):
                for k in range(len(sent)):
                    for j in range(i):
                        tag.append(sent[k + j - 1][1])
                    if (k + i) == len(sent):
                        k = len(sent)
                    print (tag)
                    count[tuple(tag)] += 1
                    tag = []

    def tcount(self, tokens):
        """Count for an k-gram for k <= n.
 
        tokens -- the k-gram tuple.
        """
        return self.count[tokens]

 
    def unknown(self, w):
        """Check if a word is unknown for the model.
 
        w -- the word.
        """
 
    """
       Todos los métodos de HMM.
    """
            







