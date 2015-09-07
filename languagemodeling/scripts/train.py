"""Train an n-gram model.

Usage:
  train.py -n <n> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""


import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import RegexpTokenizer
pattern = 
"r'''(?x)    # set flag to allow verbose regexps" +
"     ([A-Z]\.)+        # abbreviations, e.g. U.S.A." +
"   | \w+(-\w+)*        # words with optional internal hyphens" +
"   | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%" +
"   | \.\.\.            # ellipsis" +
"  | [][.,;'?():-_`]  # these are separate tokens; includes ], [" +
" '''"

tokenizer = RegexpTokenizer(pattern)
corpus = PlaintextCorpusReader('.', 'En_busca_del_tiempo_perdido.txt', word_tokenizer=tokenizer)

from docopt import docopt
import pickle

from ../languagemodeling.ngram import NGram


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    sents = corpus.sents('En_busca_del_tiempo_perdido.txt')

    # train the model
    n = int(opts['-n'])
    model = NGram(n, sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()

