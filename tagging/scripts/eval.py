"""Evaulate a tagger.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader
# from tagging.baseline import BaselineTagger


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    # tag
    hits, known, unknown = 0, 0, 0
    total, total_known, total_unknown = 0, 0 ,0 
    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)

        model_tag_sent = model.tag(word_sent)
        assert len(model_tag_sent) == len(gold_tag_sent), i

        hits_sent = []
        unknown_word = []
        known_word = []

        for j in range(len(sent)):
            equal_model_tag = (model_tag_sent[j] == gold_tag_sent[j])
            hits_sent.append(equal_model_tag)

            if model.unknown(word_sent[j]):
                unknown_word.append(equal_model_tag)
                unknown += equal_model_tag
            else:
                known_word.append(equal_model_tag)
                known += equal_model_tag

        hits += sum(hits_sent)
        total += len(sent)
        total_known += len(known_word)
        total_unknown += len(unknown_word)
        acc = float(hits) / total

        progress('{:3.1f}% ({:2.2f}%)'.format(float(i) * 100 / n, acc * 100))

    acc = float(hits) / total
    acc_known = float(known) / total_known
    acc_unknown = float(unknown) / total_unknown

    print('')
    print('Precisión: {:2.2f}%'.format(acc * 100))
    print('Precisión - palabras conocidas: {:2.2f}%'.format(acc_known * 100))
    print('Precisión - palabras desconocidas:  {:2.2f}%'.format(acc_unknown * 100))

