from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import performance_measure
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import jsonlines


def draw_confusion_matrix(matrix, labels):
    fig = plt.figure()
    #plt.gcf().canvas.set_window_title('Confusion matrix')
    #plt.title('Confusion matrix', fontsize=18)

    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest', norm=LogNorm(), cmap='viridis_r')
    fig.colorbar(cax)

    ax.set_xlabel('predicted', fontsize=16)
    ax.set_ylabel('actual', fontsize=16)
    ax.xaxis.set_label_coords(.5, -.05)

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            #color = "w" if (0 < matrix[i, j] < 10000) else "black"
            color = "gray" if (0 <= matrix[i, j] < 200) else "w"
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color=color)

    plt.show()


def ner_confusion_matrix(y_true, y_pred, draw=False):
    # convert from IOB to simple token level tags
    y_true = [tag.split('-')[1] if tag != "O" else tag for s in y_true for tag in s]
    y_pred = [tag.split('-')[1] if tag != "O" else tag for s in y_pred for tag in s]
    labels = ['PER', 'LOC', 'ORG', 'MISC', 'O']
    for l in labels:
        print(f'Number of {l} tokens in gold dataset: {y_true.count(l)}')
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    if draw:
        draw_confusion_matrix(matrix, labels)

    return matrix


def evaluate_ner(y_true, y_pred, draw=False):
    # examples
    # y_true = [[f'B-{tag}' if tag != "O" else tag for tag in s] for s in y_true]
    # y_pred = [[f'B-{tag}' if tag != "O" else tag for tag in s] for s in y_pred]

    # y_true = [['O', 'O', 'B-MISC', 'B-MISC', 'B-MISC', 'B-PER', 'O'], ['B-PER', 'B-PER', 'O']]
    # y_pred = [['O', 'O', 'B-MISC', 'B-MISC', 'B-MISC', 'B-MISC', 'O'], ['B-PER', 'B-PER', 'O']]

    print(len(y_true), len(y_pred))
    # test if predicted and true sentences have same number of tokens
    for i in range(len(y_true)):
        y = len(y_true[i])
        x = len(y_pred[i])
        if x != y:
            print(i, ':', x, y)
            print(y_pred[i])
            break

    print(f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=2))
    print(ner_confusion_matrix(y_true, y_pred, draw))
    #print(performance_measure(y_true, y_pred))


def evaluate_with_wikiann_hg(tags_pred):
    # absolute path since this script is imported to different directories
    wikiann_iob = '/home/vido/Work/school/dplm/slovak-NER/wikiann/cleaned_data_csv/test_cleaned_hg.json'

    with jsonlines.open(wikiann_iob) as f:
        true_data_sentences = list(f)

    tags_true = [sentence['word_labels'] for sentence in true_data_sentences]
    evaluate_ner(tags_true, tags_pred)


def evaluate_with_dataset(dataset_iob, tags_pred):
    with open(dataset_iob) as f:
        gold_data = f.read()

    gold_data_sentences = gold_data.split('\n\n')
    tags_true = [[line.split()[1] for line in sentence.split('\n')] for sentence in gold_data_sentences]

    evaluate_ner(tags_true, tags_pred)


def evaluate_with_gold(tags_pred):
    gold_iob = '/home/vido/Work/school/dplm/slovak-NER/final_dataset/test.txt'

    evaluate_with_dataset(gold_iob, tags_pred)


def evaluate_with_wikiann(tags_pred):
    wikiann_iob = '/home/vido/Work/school/dplm/slovak-NER/wikiann/cleaned_data/test_cleaned.txt'

    evaluate_with_dataset(wikiann_iob, tags_pred)