import spacy
from spacy.tokens import Doc
import tqdm
from spacy.vocab import Vocab


def predict(dataset, model):
    spacy.require_gpu(0)
    ner_model = spacy.load(f"./output-article/{dataset}/{model}/model-best")

    def custom_tokenizer(text):
        return Doc(ner_model.vocab, text.split(' '))

    ner_model.tokenizer = custom_tokenizer
    gold_iob = '../data/article/test-bsnlp.txt'
    out_path = f'./output-article/{dataset}/{model}/test-predictions-bsnlp.txt'

    with open(gold_iob) as f:
        gold_data = f.read()
    sentences = gold_data.split('\n\n')

    with open(out_path, 'w') as f:
        for sentence in tqdm.tqdm(sentences):
            lines = sentence.split('\n')
            tokens = [l.split()[0] for l in lines]
            tags = [l.split()[1] for l in lines]
            doc = ner_model(' '.join(tokens))
            for t in doc:
                tag = 'O' if t.ent_iob_ == 'O' else f'{t.ent_iob_}-{t.ent_type_}'
                print(t, tag, file=f)
            print('', file=f)


# x = [
#     ('wikiann', 'mdeberta'),
#     ('wikiann', 'slovakbert'),
#     ('wikiann', 'xlm-roberta'),
#     ('mix', 'mdeberta'),
#     ('mix', 'slovakbert'),
#     ('mix', 'xlm-roberta'),
# ]

x = [
    ('gold', 'mdeberta'),
    ('gold', 'slovakbert'),
    ('gold', 'xlm-roberta'),
]

for dataset, model in x:
    predict(dataset, model)
