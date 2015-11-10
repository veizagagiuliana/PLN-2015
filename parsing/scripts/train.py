"""Train a parser.

Usage:
  train.py [-m <model>] -o <file> [-n <horzMarkov>]
  train.py -h | --help

Options:
  -m <model>        Model to use [default: flat]:
                      flat: Flat trees
                      rbranch: Right branching trees
                      lbranch: Left branching trees
  -o <file>         Output model file.
  -n <horzMarkov>   Output model file.
  -h --help         Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader

from parsing.baselines import Flat, RBranch, LBranch
from parsing.upcfg import UPCFG

models = {
    'flat': Flat,
    'rbranch': RBranch,
    'lbranch': LBranch,
    'upcfg': UPCFG,
}

if __name__ == '__main__':
    opts = docopt(__doc__)

    print('Loading corpus...')
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)

    print('Training model...')

    n = opts['-n']

    if opts['-m'] == 'upcfg' and n is not None:
        model = models[opts['-m']](corpus.parsed_sents(), horzMarkov=int(n))
    else:
        model = models[opts['-m']](corpus.parsed_sents())

    print('Saving...')
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
