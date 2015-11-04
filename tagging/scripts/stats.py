"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
from collections import defaultdict

from corpus.ancora import SimpleAncoraCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/')
    sents = list(corpus.tagged_sents())
    words, tags = zip(*list(corpus.tagged_words()))

    # compute the statistics
    print('oraciones: {}'.format(len(sents)))
    print('ocurrencia de palabras: {}'.format(len(words)))
    print('vocabulario de palabras: {}'.format(len(set(words))))
    print('vocabulario de etiquetas: {}'.format(len(set(tags))))

    count_tags = defaultdict(int)
    dicttag_withword = defaultdict(int)

    for i in range(len(tags)):
        count_tags[tags[i]] += 1
        dicttag_withword[(tags[i], words[i])] += 1

    list_ord = sorted(count_tags.items(), key=lambda tup: tup[1], reverse=True)
    list_ordword = sorted(dicttag_withword.items(), key=lambda tup: tup[1],
                          reverse=True)

    print('ETIQUETAS'.rjust(8), 'CANTIDAD'.rjust(10),
          'PORCENTAJE'.rjust(20), 'PALABRAS MAS FRECUENTES'.rjust(25))
    for i in range(0, 10):
        tag, count = list_ord[i]
        repeat_words = []
        for i in range(len(list_ordword)):
            if list_ordword[i][0][0] == tag:
                repeat_words.append(list_ordword[i][0][1])
            if len(repeat_words) == 5:
                break
        print(tag.rjust(8), repr(count).rjust(10),
              (str(count/len(tags)*100) + '%').rjust(22), repeat_words)

    # compute the statistics
    print('sents: {}'.format(len(sents)))
