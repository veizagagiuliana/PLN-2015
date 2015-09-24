
"""Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""

import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import RegexpTokenizer

corpus = PlaintextCorpusReader(u'.', 'En_busca_del_tiempo_perdido.txt')

from docopt import docopt
import pickle

from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram, BackOffNGram

if __name__ == '__main__':
    opts = docopt(__doc__)

    n = int(opts['-n'])
    m = opts['-m']
    if m == 'addone':
      m = AddOneNGram
    elif m == 'interpolated':
      m = InterpolatedNGram
    elif m == 'backoff':
      m = InterpolatedNGram
    else:
      m = NGram
    sents = corpus.sents('En_busca_del_tiempo_perdido.txt')
    len_sents = len(sents)
    train_sents = sents[:int(0.9 * len_sents)]
    model = m(n, train_sents)
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
