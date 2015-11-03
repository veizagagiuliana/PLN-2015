"""
Evaluate a language model using the test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
 """

from nltk.corpus import PlaintextCorpusReader

corpus = PlaintextCorpusReader(u'.', 'En_busca_del_tiempo_perdido.txt')

from docopt import docopt
import pickle

if __name__ == '__main__':
    opts = docopt(__doc__)
    sents = corpus.sents('En_busca_del_tiempo_perdido.txt')
    len_sents = len(sents)
    eval_sents = sents[int(0.9 * len_sents):]

    filename = opts['-i']
    openfile = open(filename, 'rb')
    model = pickle.load(openfile)
    openfile.close()

    print('log_probability = ' + str(model.log_probability(eval_sents)))
    print('cross_entropy = ' + str(model.cross_entropy(eval_sents)))
    print('perplexity = ' + str(model.perplexity(eval_sents)))
