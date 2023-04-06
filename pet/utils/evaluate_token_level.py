import jsonlines
import sys

sys.path.append('../../utils')
from evaluate_ner import evaluate_ner


def evaluate_pet(test_path, predictions_path):
    with jsonlines.open(test_path, 'r') as f:
        test = list(f)

    with jsonlines.open(predictions_path, 'r') as f:
        predictions = list(f)

    tags_true = [line['tags'] for line in test]

    idxs = set(map(lambda x: x['idx'], predictions))
    tags_pred = [[y['label'] for y in predictions if y['idx'] == x] for x in idxs]

    tags_true = [[f'B-{tag}' if tag != "O" else tag for tag in s] for s in tags_true]
    tags_pred = [[f'B-{tag}' if tag != "O" else tag for tag in s] for s in tags_pred]

    evaluate_ner(tags_true, tags_pred)


def main():
    test_path = 'data/ner/test.jsonl'
    predictions_path = 'output/ner_04/final/p0-i0/predictions.jsonl'
    evaluate_pet(test_path, predictions_path)


if __name__ == '__main__':
    main()