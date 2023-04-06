import sys

sys.path.append('../../utils')
from evaluate_ner import *

def main():

    predictions_path = '/home/vido/Work/school/dplm/slovak-NER/article/transformers/output-bsnlp/xlm-roberta/predictions.txt'
    #predictions_path = '../../article/transformers/predictions.txt'
    #predictions_path = 'predictions.txt'

    with open(predictions_path) as f:
        content = f.read().strip()

    predictions_sentences = content.split('\n')
    predictions = [p_s.split() for p_s in predictions_sentences]

    #print(predictions[0])

    evaluate_with_bsnlp(predictions)
    #evaluate_with_gold(predictions)
    #evaluate_with_gold_ignore_misc(predictions)

    #evaluate_with_wikiann(predictions)
    #evaluate_with_wikiann_hg(predictions)

if __name__ == '__main__':
    main()