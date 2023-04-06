import spacy
import json
import jsonlines
import os
import re
from evaluate_ner import evaluate_ner


def get_iob(data):
    '''Get IOB tags from Prodigy/Spacy format'''
    docs = []
    # create example only if there is entity in document
    examples = ((eg["text"], eg) for eg in data if len(eg['spans']) > 0)
    nlp = spacy.blank("en")
    for doc, eg in nlp.pipe(examples, as_tuples=True):
        ents = []
        for s in eg["spans"]:
            ents.append(doc.char_span(s["start"], s["end"], s["label"], alignment_mode="expand"))
        doc.ents = ents

        iob_tags = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ != "O" else "O" for t in doc]
        docs.append({
            'tokens': [t for t in doc],
            'tags': iob_tags,
            'id': eg['meta']['id'],
        })

    return docs


def create_txt_iob_file(wikiann_bert_dir, output_path):
    '''Create txt IOB dataset file from batches of inferences of transformer
    created during pre-annotation phase
    '''
    # load all inference documents
    all_docs = []
    for root, dirs, files in os.walk(wikiann_bert_dir):
        for name in sorted(files):
            with jsonlines.open(os.path.join(wikiann_bert_dir, name), 'r') as bert_file:
                print(os.path.join(wikiann_bert_dir, name))
                data = list(bert_file)
                docs = get_iob(data)
                all_docs.extend(docs)

    # create txt format IOB dataset
    with open(output_path, 'w') as f:
        for doc in all_docs:
            # if len([t for t in doc['tags'] if t != 'O']) == 0:
            #     continue

            # rejected articles or articles with no entities after manual anntoation
            if doc['id'] in ["200651", "664440", "10410", "81675", "95198", "249703", "276758",
                             "358743", "369401", "623007", "73922", "233613", "309103", "491524",
                              "499259", "516628", "664665", "92489", "125292", "593924", "54777",
                             "94303", "77496", "173384", "81146", "668055", "9892", "159602",
                             ]:
                continue
            for token, tag in zip(doc['tokens'], doc['tags']):
                if str(token) == '\n\t':
                    print('\n', end='', file=f)
                else:
                    print(token, tag, file=f)
            print('\n', end='', file=f)


def evaluate(wikiann_bert_iob, gold_iob):
    '''Evaluate results of transformer model trained on WikiANN data with
    manually created gold dataset.
    '''
    with open(wikiann_bert_iob) as f:
        wikiann_bert_data = f.read()

    with open(gold_iob) as f:
        gold_data = f.read()

    wikiann_bert_sentences = wikiann_bert_data.split('\n\n')
    gold_data_sentences = gold_data.split('\n\n')

    tags_pred = [ [ line.split()[1] for line in sentence.split('\n')] for sentence in wikiann_bert_sentences ]
    tags_true = [ [ line.split()[1] for line in sentence.split('\n')] for sentence in gold_data_sentences ]
    # remove MISC tags from dataset
    #tags_true = [ [ line.split()[1] if line.split()[1].find("MISC") == -1 else 'O' for line in sentence.split('\n')] for sentence in gold_data_sentences ]

    evaluate_ner(tags_true, tags_pred, draw=True)


def main():
    wikiann_bert_dir = '../prodigy/wikiann_bert'
    wikiann_bert_txt_iob = 'wikiann_bert_iob.txt'

    gold_iob = '../prodigy/final_txt/dataset.uncleared.txt'

    #create_txt_iob_file(wikiann_bert_dir, wikiann_bert_txt_iob)
    evaluate(wikiann_bert_txt_iob, gold_iob)


if __name__ == '__main__':
    main()
