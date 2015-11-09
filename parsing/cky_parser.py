from collections import defaultdict
from nltk.tree import Tree


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """

        self._pi = {}
        self._bp = {}
        self.grammar = grammar
        self.lexical = lexical = defaultdict(list)
        self.nonlexical = nonlexical = defaultdict(list)
       
        for elem in grammar.productions():
            if elem.is_lexical():
                rhs = elem.rhs()[0]
                lexical[rhs] += [elem]
            else:
                rhs = elem.rhs()
                nonlexical[(rhs[0].symbol(), rhs[1].symbol())] += [elem]

    def parse(self, sent):
        """Parse a sequence of terminals.
        sent -- the sequence of terminals.
        # """
        self._pi = pi = defaultdict(dict)
        self._bp = bp = defaultdict(dict)
        n = len(sent)
        found = False
        lex = self.lexical
        nonlex = self.nonlexical

        for i, word in enumerate(sent, 1):
            prod = lex[word]
            for p in prod:
                non_ter = p.lhs().symbol()
                log_prob = p.logprob()
                pi[(i, i)][non_ter] = log_prob
                bp[(i, i)][non_ter] = Tree(non_ter, [word])

        for k in range(1, n):
            for j in range(1, n-k+1):
                s = j + k
                # pi[(j, s)] = {}
                # bp[(j, s)] = {}
                for q in range(j, s):
                    rama_i = pi[(j, q)]
                    rama_d = pi[(q+1, s)]
                    p = []
                    for elem1 in rama_i:
                        for elem2 in rama_d:
                            p += nonlex[(elem1, elem2)]
                    for elem in p:
                        rhs = elem.rhs()
                        izq =rhs[0].symbol()
                        der =rhs[1].symbol()
                        # if izq in rama_i and der in rama_d:
                        non_ter = repr(elem.lhs())
                        log_prob = elem.logprob() + rama_i[izq] + rama_d[der]
                        child_izq = bp[(j, q)][izq]
                        child_der = bp[(q+1, s)][der]
                        if non_ter in pi[(j, s)]:
                            if log_prob > pi[(j, s)][non_ter]:
                                pi[(j, s)][non_ter] = log_prob
                                bp[(j, s)][non_ter] = Tree(non_ter,
                                                      [child_izq, child_der])
                        else:
                            pi[(j, s)][non_ter] = log_prob
                            bp[(j, s)][non_ter] = Tree(non_ter,
                                                       [child_izq, child_der])
                        found = True
                if found:
                    found = False
                else:
                    pi[(j, s)] = {}
                    bp[(j, s)] = {}

        start = self.grammar.start().symbol()
        lp = 0.0
        t = None
        if start in bp[(1, n)]:
            lp = pi[(1, n)][start]
            t = bp[(1, n)][start]
        return(lp, t)
