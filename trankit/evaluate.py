import pickle
import sys

sys.path.append('../utils')
from evaluate_ner import evaluate_with_gold, evaluate_with_wikiann


def main():
    predictions_path = 'predictions'

    with open(predictions_path, "rb") as fp:
        predictions = pickle.load(fp)

    #evaluate_with_gold(predictions)
    evaluate_with_wikiann(predictions)


if __name__ == '__main__':
    main()