"""Train a sequence tagger.

Usage:
  train.py [-n <n>] [-c <c>] [-m <model>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -c <c>        Sorter of the model.
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
from tagging.memm import MEMM

models = {
    'base': BaselineTagger,
    'mlhmm': MLHMM,
    'memm' : MEMM,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    # train the model
    exit = False
    if models[opts['-m']] == MLHMM:
      model = models[opts['-m']](int(opts['-n']), sents) 
    elif models[opts['-m']] == MEMM:
      if opts['-c'] == 'LR' or opts['-c'] =='MNB' or opts['-c'] =='LSVC':
        model = models[opts['-m']](int(opts['-n']), sents, opts['-c']) 
      else:
        print('Parametro -c incorrecto.')
        print('Ingrese alguna de las siguientes clasificaciones:')
        print(' LR: LogisticRegression\n',
              'MNB: MultinomialNB\n',
              'LSVC: LinearSVC\n')
        exit = True
    else: 
      model = models[opts['-m']](sents)

    # save it
    if not exit:
      filename = opts['-o']
      f = open(filename, 'wb')
      pickle.dump(model, f)
      f.close()
