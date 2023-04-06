import re


def process_row(row, conll_2003=False):
    cleaned = re.sub(r'^sk:', '', row)
    splitted = cleaned.split()
    if len(splitted) <= 0:
        return
    # text = re.sub(r'[^\w\s.,!?]', '', splitted[0])
    text = splitted[0]
    tag = splitted[1]
    if len(text) > 0:
        return tag


def count_wiki(in_path, c):
    '''Count number of entities'''
    with open(in_path) as in_f:
        content = in_f.read()
        for chunk in content.split('\n\n'):
            beg = None
            for row in chunk.split('\n'):
                tag = process_row(row)
                if tag is not None:
                    if tag == 'B-ORG':
                        c['O'] += 1
                    elif tag == 'B-PER':
                        c['P'] += 1
                    elif tag == 'B-LOC':
                        c['L'] += 1


def main():
    datasets = ['test', 'dev', 'train']
    c = {'L':0, 'P': 0, 'O': 0}
    for dataset in datasets:
        c = {'L': 0, 'P': 0, 'O': 0}
        in_path = f'raw_data/{dataset}.txt'
        count_wiki(in_path, c)
        print(c)


if __name__ == "__main__":
    main()
