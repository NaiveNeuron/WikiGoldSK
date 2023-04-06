import re
import collections


def count_entities(dataset_path):

    with open(dataset_path, 'r') as f:
        data = f.read()

    entities = {'PER': 0, 'LOC': 0, 'ORG': 0, 'MISC': 0}
    entity_pattern = r"B-{}\n"

    for entity in entities.keys():
        entity_re = entity_pattern.replace('{}', entity)
        entities[entity] = len(list(re.findall(entity_re, data, re.M)))

    print(f'Number of entities per class: {entities}')
    print('Number of entities:', sum([n for _, n in entities.items()]))


def count_tokens(dataset_path):
    with open(dataset_path, 'r') as f:
        data = f.read()

    data = re.sub(r'\n\n', r'\n', data)

    tokens = [line.split()[0].lower() for line in data.split("\n")]
    counter = collections.Counter(tokens)

    print(f'Number of tokens in {dataset_path} are: {len(tokens)}')
    print(f'Number of unique tokens in {dataset_path} are: {len(set(tokens))}')
    print(f'100 most common tokens in {dataset_path} are: {counter.most_common(100)}')


def longest_sentences(dataset_path):
    with open(dataset_path, 'r') as f:
        data = f.read()

    sentences = data.split('\n\n')
    sentences = [' '.join([line.split()[0] for line in s.split('\n')]) for s in sentences]
    print(sentences[:5])
    sentences.sort(key=lambda x: len(x), reverse=True)
    print('Length of 5 longest sentences:', [len(s) for s in sentences[:5]])
    print(f'Number of sentences longer than 512 chars/number of all sentences: {sum([1 if len(s) > 512 else 0 for s in sentences])}/{len(sentences)}')


def main():
    dataset_path = '../final_dataset/dataset.txt'

    count_entities(dataset_path)
    count_tokens(dataset_path)
    longest_sentences(dataset_path)


if __name__ == '__main__':
    main()
