
from collections import defaultdict

class BaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        self.tagged_tokens = tagged_tokens = {}

        count = defaultdict(int)
        for sent in tagged_sents:
            for token, tagged in sent:
                tagged_tokens[token] = tagged
                count[tagged] += 1
        more_common = sorted(count.items(), key=lambda tup: tup[1], reverse=True)
        self.more_common = more_common[0][0]

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
            return self.most_common
        else:
            return self.tagged_tokens[w]


    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return (w not in self.tagged_tokens)

