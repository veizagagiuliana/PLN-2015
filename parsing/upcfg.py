from collections import defaultdict
from nltk.tree import Tree
from nltk.grammar import Nonterminal as N, ProbabilisticProduction
from nltk import induce_pcfg
from .cky_parser import CKYParser
from .util import lexicalize


class UPCFG:
    """Unlexicalized PCFG.
    """
 
    def __init__(self, parsed_sents, start='sentence'):
        """
        parsed_sents -- list of training trees.
        """
        self.parsed_sents = parsed_sents
        self.start = N(start)
        self._productions = productions = parsed_sents[0].productions()

        self.count = count = defaultdict(dict)

        for prod in productions:
            lhs = prod.lhs()
            rhs = prod.rhs()
            if not lhs in count:
                count[lhs][lhs] = 0
                count[lhs][rhs] = 0
            if not rhs in count[lhs]:
                count[lhs][rhs] = 0
            count[lhs][lhs] += 1
            count[lhs][rhs] += 1

        self._grammar = induce_pcfg(N(start), productions)
        self.CKY = CKYParser(self._grammar)

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        result = []
        productions = self._productions
        for prod in productions:
            lhs = prod.lhs()
            rhs = prod.rhs()
            prob_rhs = float(self.count[lhs][rhs] / self.count[lhs][lhs])
            if not prod.is_lexical():
                result += [ProbabilisticProduction(lhs,
                                                   [rhs[0], rhs[1]],
                                                    prob=prob_rhs)]
            else:
                result += [ProbabilisticProduction(lhs, [lhs.symbol()],
                                                    prob=1.0)]

        # return self._grammar.productions()
 
    def parse(self, tagged_sent):
        """Parse a tagged sentence.
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        start = repr(self.start)
        l = []
        words, tags = zip(*tagged_sent)
        log, tree = self.CKY.parse(words)

        if tree is not None:
            result = lexicalize(tree, words)
        else:
            tree = [Tree(tag, [word]) for word, tag in tagged_sent]
            print('has')
            result = Tree(start, tree)
        print(result)
        return result


