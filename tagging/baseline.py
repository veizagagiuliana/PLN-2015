from operator import itemgetter
from collections import defaultdict


class BaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        tagged_tokens = defaultdict(dict)
        self.word_tag = word_tag = defaultdict(int)

        count = defaultdict(int)
        for sent in tagged_sents:
            for token, tagged in sent:
                count[tagged] += 1
                if tagged not in tagged_tokens[token]:
                    tagged_tokens[token][tagged] = 1
                else:
                    tagged_tokens[token][tagged] += 1

        for word, tag in tagged_tokens.items():
            word_tag[word] = max(tag.items(), key=itemgetter(1))[0]

        more_common = max(count.items(), key=itemgetter(1))[0]
        self.more_common = more_common

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        if self.unknown(w):
            return self.more_common
        else:
            return self.word_tag[w]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return (w not in self.word_tag)
