from nltk.tree import Tree
from nltk.grammar import Nonterminal as N
from nltk import induce_pcfg
from .cky_parser import CKYParser
from .util import unlexicalize, lexicalize


class UPCFG:
    """Unlexicalized PCFG.
    """

    def __init__(self, parsed_sents, start='sentence', horzMarkov=None):
        """
        parsed_sents -- list of training trees.
        """
        self.start = start = N(start)
        productions = []

        for tree in parsed_sents:
            unlex = unlexicalize(tree.copy(deep=True))
            unlex.chomsky_normal_form(horzMarkov=horzMarkov)
            unlex.collapse_unary(collapsePOS=True, collapseRoot=True)
            productions += unlex.productions()

        self._grammar = grammar = induce_pcfg(start, productions)
        self.CKY = CKYParser(grammar)

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self._grammar.productions()

    def parse(self, tagged_sent):
        """Parse a tagged sentence.
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        start = repr(self.start)
        sent, tags = zip(*tagged_sent)
        log, tree = self.CKY.parse(tags)

        if tree is not None:
            tree.un_chomsky_normal_form()
            result = lexicalize(tree, sent)
        else:
            tree = [Tree(tag, [word]) for word, tag in tagged_sent]
            result = Tree(start, tree)
        return result
