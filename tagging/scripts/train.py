"""Train a sequence tagger.

Usage:
  train.py [-n <n>] [-m <model>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: base]:
                  base: Baseline
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger
from tagging.hmm import MLHMM

models = {
    'base': BaselineTagger,
    'mlhmm': MLHMM,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    # train the model

    if models[opts['-m']] == MLHMM:
      model = models[opts['-m']](int(opts['-n']), sents) 
    else: 
      model = models[opts['-m']](sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
