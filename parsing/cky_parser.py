from collections import defaultdict
from nltk.tree import Tree
from math import log2


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """

        self.grammar = grammar
        self.lex = lex = []
        self.notlex = notlex = []
        self._pi = {}

        for elem in grammar.productions():
            if elem.is_lexical():
                lex += [elem]
            else:
                notlex += [elem]

    def parse(self, sent):
        """Parse a sequence of terminals.
        sent -- the sequence of terminals.
        # """
        self._pi = pi = defaultdict(dict)
        self._bp = bp = defaultdict(dict)
        n = len(sent)
        found = False
        lex = self.lex
        notlex = self.notlex

        for i, word in enumerate(sent, 1):
            for elem in lex:
                rhs = elem.rhs()
                for k in range(len(rhs)):
                    if word == rhs[k]:
                        non_ter = repr(elem.lhs())
                        log_prob = log2(elem.prob())
                        pi[(i, i)] = {non_ter: log_prob}
                        bp[(i, i)] = {non_ter: Tree(non_ter, [word])}
                        found = True
                        break
                if found:
                    found = False
                    break

        for k in range(1, n):
            for j in range(1, n-k+1):
                s = j + k
                for q in range(j, s):
                    rama_i = pi[(j, q)]
                    rama_d = pi[(q+1, s)]
                    for elem in notlex:
                        rhs = elem.rhs()
                        izq = repr(rhs[0])
                        der = repr(rhs[1])
                        if izq in rama_i and der in rama_d:
                            non_ter = repr(elem.lhs())
                            log_prob = (log2(elem.prob()) + rama_i[izq] +
                                        rama_d[der])
                            child_izq = bp[(j, q)][izq]
                            child_der = bp[(q+1, s)][der]
                            if non_ter in pi[(j, s)]:
                                if log_prob > pi[(j, s)][non_ter]:
                                    pi[(j, s)] = {non_ter: log_prob}
                                    bp[(j, s)] = {non_ter: Tree(non_ter,
                                                  [child_izq, child_der])}
                            else:
                                pi[(j, s)] = {non_ter: log_prob}
                                bp[(j, s)] = {non_ter: Tree(non_ter,
                                              [child_izq, child_der])}
                            found = True
                            break

                if found:
                    found = False
                else:
                    bp[(j, s)] = {}

        start = repr(self.grammar.start())
        lp = 0.0
        t = {}
        if start in bp[1, n]:
            lp = pi[(1, n)][start]
            t = bp[(1, n)][start]

        return(lp, t)
