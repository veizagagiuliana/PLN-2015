"""
Evaulate a parser.

Usage:
  eval.py -i <file> [-m <m>] [-n <n>]
  eval.py -h | --help

Options:
  -i <file>     Parsing model file.
  -m <m>        Parse only sentences of length <= <m>.
  -n <n>        Parse only <n> sentences (useful for profiling).
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader

from parsing.util import spans


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    print('Loading model...')
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    if opts['-m'] != None:
        m = int(opts['-m'])
        no_m = False
    else:
        no_m = True

    print('Loading corpus...')
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    parsed_sents = list(corpus.parsed_sents())

    print('Parsing...')
    hits, total_gold, total_model = 0, 0, 0
    uhits, utotal_gold, utotal_model = 0, 0, 0

    if opts['-n'] != None:
        n = int(opts['-n'])
    else:
        n = len(parsed_sents)

    format_str = '{:3.1f}% ({}/{}) Labeled: (P={:2.2f}%, R={:2.2f}%, F1={:2.2f}%) UnLabeled: (P={:2.2f}%, R={:2.2f}%, F1={:2.2f}%)'
    progress(format_str.format(0.0, 0, n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    prec, rec, f1 = 0, 0, 0
    uprec, urec, uf1 = 0, 0, 0


    for i, gold_parsed_sent in enumerate(parsed_sents[:n]):
        tagged_sent = gold_parsed_sent.pos()
        if m >= len(tagged_sent) or no_m:
            # parse
            model_parsed_sent = model.parse(tagged_sent)

            # compute labeled scores
            gold_spans = spans(gold_parsed_sent, unary=False)
            model_spans = spans(model_parsed_sent, unary=False)
            hits += len(gold_spans & model_spans)
            total_gold += len(gold_spans)
            total_model += len(model_spans)

            # compute labeled partial results
            prec = float(hits) / total_model * 100
            rec = float(hits) / total_gold * 100
            f1 = 2 * prec * rec / (prec + rec)

            # compute unlabeled scores

            ugold_spans = {(y,z) for x, y, z in gold_spans}
            umodel_spans = {(y,z) for x, y, z in model_spans}
            uhits += len(ugold_spans & umodel_spans)
            utotal_gold += len(ugold_spans)
            utotal_model += len(umodel_spans)

            # compute labeled partial results
            uprec = float(uhits) / utotal_model * 100
            urec = float(uhits) / utotal_gold * 100
            uf1 = 2 * uprec * urec / (uprec + urec)

        progress(format_str.format(float(i+1) * 100 / n, i+1, n, prec, rec, f1,
                                    uprec, urec, uf1))

    print('')
    print('Parsed {} sentences'.format(n))
    print('Labeled')
    print('  Precision: {:2.2f}% '.format(prec))
    print('  Recall: {:2.2f}% '.format(rec))
    print('  F1: {:2.2f}% '.format(f1))
    print('Unlabeled')
    print('  Precision: {:2.2f}% '.format(uprec))
    print('  Recall: {:2.2f}% '.format(urec))
    print('  F1: {:2.2f}% '.format(uf1))
