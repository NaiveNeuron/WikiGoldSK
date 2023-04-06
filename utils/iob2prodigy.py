from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.training import biluo_tags_to_offsets
from spacy.training import iob_to_biluo
import textspan
import jsonlines
from bs4 import BeautifulSoup as BSHTML


def find_token_index(span_position, tokens):
    for i, token in enumerate(tokens):
        if token['start'] <= span_position < token['end']:
            return i
    return -1


def wikidump_to_prodigy(wikidump_labeled, output_path):
    docs = []

    print('CONVERTING SLOVAK BERT WIKI ANNOTATIONS TO PRODIGY FORMAT')
    for doc in wikidump_labeled:
        tokens = []
        spans = []
        sentences = ''

        for i, sentence in enumerate(doc['sentences']):

            tokenized_sentence = sentence['text'].split(' ')
            tokens_spans = textspan.get_original_spans(tokenized_sentence, sentence['text'])
            tokens.extend([{
                'id': j + len(tokens),
                'text': text,
                'start': spans[0][0] + len(sentences),
                'end': spans[0][1] + len(sentences),
            } for j, (text, spans) in enumerate(zip(tokenized_sentence, tokens_spans))])

            sentence_spans = []
            for ent in sentence['ents']:

                # fix code after problem with "token_end: -1" (when inference span was with whitespace)
                if ent['text'].strip() != ent['text']:
                    forward_blank_len = len(ent['text'].lstrip()) - len(ent['text'])
                    backward_blank_len = len(ent['text'].rstrip()) - len(ent['text'])
                    ent['text'] = ent['text'].strip()
                    ent['start'] -= forward_blank_len
                    ent['end'] -= backward_blank_len

                sentence_spans.append({
                    'text': ent['text'].strip(),
                    'label': ent['label'],
                    'start': ent['start'] + len(sentences),
                    'end': ent['end'] + len(sentences),
                    'token_start': find_token_index(ent['start'] + len(sentences), tokens),
                    'token_end': find_token_index(ent['end'] + len(sentences) - 1, tokens),
                })
            spans.extend(sentence_spans)
            # spans.extend([{
            #     'text': ent['text'],
            #     'label': ent['label'],
            #     'start': ent['start'] + len(sentences),
            #     'end': ent['end'] + len(sentences),
            #     'token_start': find_token_index(ent['start'] + len(sentences), tokens),
            #     'token_end': find_token_index(ent['end'] + len(sentences) - 1, tokens, [ent['text'], ent['end'], len(sentences) - 1]),
            # } for ent in sentence['ents']])

            sentence_separator = ''
            if i != len(doc['sentences']) - 1:
                sentence_separator = '\n\t'
                tokens.append({
                    'id': len(tokens),
                    'text': sentence_separator,
                    'start': len(sentences) + len(sentence['text']),
                    'end': len(sentences) + len(sentence['text']) + len(sentence_separator),
                })

            sentences += sentence['text'] + sentence_separator

        doc_prodigy = {
            'text': sentences,
            'spans': spans,
            'tokens': tokens,
            'meta': doc['meta'],
        }
        docs.append(doc_prodigy)

    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(docs)


def load_dataset(path):
    output = []
    with open(path) as f:
        data = f.read()
        BS = BSHTML(data)
        docs_in_file = BS.findAll("doc")
        # docs_texts = [ doc.getText for doc in docs_in_file ]
        # ids.extend([doc['id'] for doc in docs_in_file if is_doc_valid(doc.getText())])
        for doc in docs_in_file:
            sentences = doc.getText().strip().split('\n\n')
            splitted_sentences = []
            for sentence in sentences:
                splitted_sentences.append((
                    [line.split()[0] for line in sentence.split('\n')],
                    [line.split()[1] for line in sentence.split('\n')],
                ))
            output.append({"id": doc['id'], "sentences": splitted_sentences})

    return output


def dataset_to_prodigy(path):
    docs = load_dataset(path)
    converted_docs = []
    #raw_sentences = load_dataset('test-dataset.txt')
    for doc in docs:
        sentences = []
        for tokens, tags in doc['sentences']:
            text = ' '.join(tokens)
            biluo_tags = iob_to_biluo(tags)

            biluo_doc = Doc(Vocab(), words=tokens)
            offsets = biluo_tags_to_offsets(biluo_doc, biluo_tags)

            sentences.append(
                {
                    'text': text,
                    "ents": [
                        {'text': text[offset[0]:offset[1]], 'label': offset[2], 'start': offset[0], 'end': offset[1]}
                        for offset in offsets
                    ]
                }
            )
        converted_docs.append(
            {
                "sentences": sentences,
                "meta": {"id": int(doc['id'])}
            }
        )

    wikidump_to_prodigy(
        converted_docs,
        "test-prodigy-input.jsonl"
    )


dataset_to_prodigy('../final_dataset/dataset_docs.txt')
#dataset_to_prodigy('test-dataset.txt')


# sentence=[
#     "Eswatini",
#     ",",
#     "dlhý",
#     "tvar",
#     "Eswatinské",
#     "kráľovstvo",
#     "(",
#     "do",
#     "roku",
#     "2018",
#     "[",
#     "na",
#     "Slovensku",
# ]
#
# tags = ["B-LOC", "O", "O", "O", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O", "B-LOC"]
# biluo_tags = iob_to_biluo(tags)
#
#
# doc = Doc(Vocab(), words=sentence)
# offsets = biluo_tags_to_offsets(doc, biluo_tags)
# print(offsets)
#
# wikidump_to_prodigy(
#     [{
#         "sentences": [{
#             'text': ' '.join(sentence),
#             "ents": [
#                 {'text': ' '.join(sentence)[offset[0]:offset[1]], 'label': offset[2], 'start': offset[0], 'end': offset[1]} for offset in offsets
#             ]
#         }],
#         "meta": {"id": 1}
#
#     }],
#     "test.txt"
# )



