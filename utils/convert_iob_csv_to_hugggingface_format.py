import pandas as pd


def convert(in_path, out_path):
    '''Convert IOB data in csv format to hugging face format'''

    data = pd.read_csv(in_path, encoding='utf-8')

    data['sentence_new'] = data[['sentence','word','tag']].groupby(['sentence'])['word'].apply(tuple)
    data['word_labels'] = data[['sentence','word','tag']].groupby(['sentence'])['tag'].apply(tuple)

    data = data[["sentence_new", "word_labels"]].drop_duplicates().reset_index(drop=True)
    data = data.iloc[1:]
    data['sentence'] = data['sentence_new'].apply(list)
    data['word_labels'] = data['word_labels'].apply(list)
    data = data.drop('sentence_new', 1)

    data.to_json(out_path, orient='records', lines=True, force_ascii=False)


def main():
    datasets = ['test', 'dev', 'train']
    for dataset in datasets:
        in_path = f'{dataset}.csv'
        out_path = f'{dataset}_hg.json'
        convert(in_path, out_path)


if __name__ == "__main__":
    main()