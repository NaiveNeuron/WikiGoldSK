import re
import csv

SENTENCE_ID = 0


def process_row(row, conll_2003=False):
    global SENTENCE_ID

    # remove 'sk' prefix from wikiann datasets
    cleaned = re.sub(r'^sk:', '', row)
    splitted = cleaned.split()
    if len(splitted) <= 0:
        return
    # clean row with regex
    #text = re.sub(r'[^\w\s.,!?]', '', splitted[0])
    text = splitted[0]

    tag = splitted[1]
    if len(text) > 0:
        if conll_2003:
            # added POS tag
            return [str(SENTENCE_ID), text, 'O', tag]
        else:
            return [str(SENTENCE_ID), text, tag]


def convert(in_path, out_path):
    '''Convert IOB from txt format to following csv format:
        sentence,word,pos,tag
        <sentence_id>,<word>,O,<iob_tag>
    '''
    global SENTENCE_ID

    with open(in_path) as in_f:
        with open(out_path, 'w') as out_f:
            csvWriter = csv.writer(out_f)
            csvWriter.writerow(['sentence','word','pos','tag'])
            content = in_f.read()

            for sentence in content.split('\n\n'):
                SENTENCE_ID += 1
                for row in sentence.split('\n'):
                    cleaned = process_row(row, conll_2003=True)
                    if cleaned is not None:
                        csvWriter.writerow(cleaned)


def main():
    global SENTENCE_ID
    datasets = ['test', 'dev', 'train']
    for dataset in datasets:
        SENTENCE_ID = 0
        # WIKIANN
        # in_path = f'../raw_data/{dataset}.txt'
        # out_path = f'{dataset}_cleaned.csv'

        # MY_DATASET
        in_path = f'./final_dataset/{dataset}.txt'
        out_path = f'./my_dataset/transformers/{dataset}.csv'

        convert(in_path, out_path)


if __name__ == "__main__":
    main()
