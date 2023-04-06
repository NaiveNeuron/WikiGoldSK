import sys

sys.path.append('../../utils')
from evaluate_ner import *


def main():
    #predictions_path = 'output-bsnlp/slovakbert/test-predictions-bsnlp.txt'
    predictions_path = '../trankit/output/predictions-bsnlp.txt'

    with open(predictions_path) as f:
        content = f.read().strip()

    predictions_sentences = content.split('\n\n')
    predictions = [[l.split()[1] for l in p_s.split('\n')] for p_s in predictions_sentences]

    evaluate_with_bsnlp(predictions)
    #evaluate_with_wikiann(predictions)
    #evaluate_with_wikiann_hg(predictions)
    #evaluate_with_gold_ignore_misc(predictions)


if __name__ == '__main__':
    main()