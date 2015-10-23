from collections import defaultdict
from featureforge.vectorizer import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from tagging.features import *

class MEMM:
 
    def __init__(self, n, tagged_sents, clas='LR'):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self.n = n
        self.tagged_sents = tagged_sents
        self.vocabulary = vocabulary = set()

        for tagged_sent in tagged_sents:
            if tagged_sent != []:
                sent, tag = zip(*tagged_sent)
                for word in sent:
                    vocabulary.add(word)

        features = (word_lower, word_istitle, word_isupper, word_isdigit)

        for elem in features:
            features += (PrevWord(elem),)
        for i in range(1, n):
            features += (NPrevTags(i),)

        vect = Vectorizer(features)
        clf = {'LR': LogisticRegression(),
               'MNB': MultinomialNB(),
               'LSVC': LinearSVC()
               }

        self.text_clf = Pipeline([('vect', vect),
                                  ('clf', clf[clas])])
        sents_histories = self.sents_histories(tagged_sents)
        sents_tags = self.sents_tags(tagged_sents)
        self.text_clf = self.text_clf.fit(sents_histories, sents_tags)

 
    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.
 
        tagged_sents -- the corpus (a list of sentences)
        """
        history = []
        for tagged_sent in tagged_sents:
            if tagged_sent != []:
                history += self.sent_histories(tagged_sent)
        return history
        
 
    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        n = self.n
        sent, tags = zip(*tagged_sent)
        tags = ('<s>',) * (n-1) + tags
        history = []

        for i in range(len(tags)-n+1):
            history.append(History(list(sent), tags[i:i+n-1], i))

        return history
 
    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.
 
        tagged_sents -- the corpus (a list of sentences)
        """
        tags = []
        for tagged_sent in tagged_sents:
            if tagged_sent != []:
                tags += self.sent_tags(tagged_sent)
        return tags
 
    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        sent, tags = zip(*tagged_sent)
        return tags

    def tag(self, sent):
        """Tag a sentence.
 
        sent -- the sentence.
        """
        n = self.n
        prev_tags = ('<s>',) * (n-1)
        tags = []
        for i, word in enumerate(sent):
            history = History(sent, prev_tags, i)
            tag = self.tag_history(history)[0]
            tags = tags + [tag]
            prev_tags = (prev_tags + (tag,))[1:]
        return tags
 
    def tag_history(self, h):
        """Tag a history.
 
        h -- the history.
        """
        return self.text_clf.predict([h])
 
    def unknown(self, w):
        """Check if a word is unknown for the model.
 
        w -- the word.
        """
        if w in self.vocabulary:
            return False
        return True

