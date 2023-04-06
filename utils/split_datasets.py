import numpy as np
import pandas as pd


# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def load_txt_iob_data(txt_file):

    with open(txt_file) as in_f:
        content = in_f.read()

    sentences = content.split('\n\n')
    return sentences


def df_to_txt_iob(df, output_path):

    with open(output_path, 'w') as f:
        df = df.reset_index()
        for index, row in df.iterrows():
            print(row['sentences'], file=f)
            print(file=f)


def main():
    np.random.seed([42, 42])
    dataset_path = '../prodigy/final_txt/dataset3.txt'
    dataset_sentences = load_txt_iob_data(dataset_path)
    df = pd.DataFrame({ 'sentences': dataset_sentences})
    train, dev, test = train_validate_test_split(df)

    df_to_txt_iob(train, 'final_dataset/train.txt')
    df_to_txt_iob(dev, 'final_dataset/dev.txt')
    df_to_txt_iob(test, 'final_dataset/test.txt')


if __name__ == '__main__':
    main()
