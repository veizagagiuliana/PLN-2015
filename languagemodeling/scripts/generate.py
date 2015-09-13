"""
Generate natural language sentences using a language model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle
import pdb
from languagemodeling.ngram import NGram, NGramGenerator

if __name__ == '__main__':

    opts = docopt(__doc__)

    n = int(opts['-n'])

    filename = opts['-i']
    openfile = open(filename, "rb")

    model = pickle.load(openfile)
    openfile.close()
    new_sents = NGramGenerator(model)

    for k in range(n):
      generate = new_sents.generate_sent()
      print(" ".join(generate) + '\n')    
