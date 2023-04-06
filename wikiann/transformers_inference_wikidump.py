from transformers import pipeline
from spacy import displacy
import nltk
import jsonlines
import textspan
from tqdm import tqdm


nltk.download('punkt')


def visualize_ner(ner_data, out_path='visualized_ner.html'):
    '''Generates HTML from NER data'''
    html = displacy.render(ner_data, options={'colors': {'PER': 'yellow', 'MISC': '#f89ff8'}}, manual=True, style="ent", page=True)
    with open(out_path, 'w') as f:
        f.write(html)


def visualize_prodigy(annotations_prodigy):
    '''Visualize Spacy/Prodigy NER data'''
    annotations_prodigy['ents'] = annotations_prodigy['spans']
    visualize_ner(annotations_prodigy, out_path='visualized_ner_prodigy.html')


def tokenize_nltk(text):
    '''Tokenize article to sentences and words'''
    tokenized = []
    sentence_tokenizer = nltk.data.load(f"tokenizers/punkt/czech.pickle")
    extra_abbreviations = ['napr', 'lat', 'rod', 'mr', 'sv', 'mgr',
                           'prof', 'ing', 'bc', 'dr', 'rus', 'tzv', 'phd', 'drsc', 'phdr',
                           'dr', 'iii', 'ii', 'i', 'iv', 'odd', 'angl', 'skr', 'stor',
                           'pol', 'vz', 'tal', 'rndr', 'fr', 'odd', 'mad', 'var', 'grec',
                           'gr', 'nem', 'lat', 'hebr', 'arab', 'novoheb', ]
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
    sentences = sentence_tokenizer.tokenize(text)

    for sentence in sentences:
        tokenized.append(nltk.tokenize.word_tokenize(sentence, language='czech', preserve_line=True))
    return tokenized


def tokenize_split(text):
    sentences = text.split(' . ')
    return [sentence.split(' ') for sentence in sentences]


def join_tokenized_nltk(tokenized_text):
    return ' '.join([' '.join(sentence) for sentence in tokenized_text])


def connect_annotations(annotations, sentence):
    '''Map and align annotations from transformer model with tokens from NLTK'''
    annotations_s = sorted(annotations, key=lambda x: x['start'])
    connected = []
    tokens = sentence.split(' ')

    annotation_ix = 0
    token_ix = -1
    tokens_len = -1
    tokens_j = ''

    while annotation_ix < len(annotations_s):
        current_annotation = annotations_s[annotation_ix]

        while tokens_len < current_annotation['end']:
            token_ix += 1
            tokens_len += len(tokens[token_ix]) + 1
            tokens_j = tokens_j + tokens[token_ix] + ' '

        if current_annotation['end'] != tokens_len:

            new_annotation = dict(current_annotation)

            while True:
                annotation_ix += 1
                if annotation_ix >= len(annotations_s):
                    break
                current_annotation = annotations_s[annotation_ix]

                new_annotation = {
                    'text': sentence[new_annotation['start']:current_annotation['end']],
                    'label': new_annotation['label'],
                    'start': new_annotation['start'],
                    'end': current_annotation['end'],
                }

                if tokens_len <= current_annotation['end']:
                    break

            connected.append(new_annotation)
        else:
            connected.append(current_annotation)

        annotation_ix += 1

    return connected


def filter_annotations(annotations, tokenized_sentence):
    '''Filter out annotations that are mostly mistakes'''
    # remove 'headers like' annotations - one word and dot
    if len(tokenized_sentence) == 2 and tokenized_sentence[1] == '.':
        return []

    # remove annotations that don't start with capital letter
    return [ann for ann in annotations if ann['text'][0].isupper()]


def get_clear_annotations(annotations_encoded, text):
    '''Transform annotations from transformer model to more clear dict format'''
    annotations = []
    for annotation in annotations_encoded:
        annotations.append({
            'text': text[annotation['start']:annotation['end']],
            'label': annotation['entity_group'],
            'start': annotation['start'],
            'end': annotation['end'],
        })
    return annotations


def inference(pipeline, tokenized_sentence):
    '''Predict, transform and filter NER labels for a sentence'''
    joined_sentence = ' '.join(tokenized_sentence)
    annotations = pipeline(joined_sentence)
    clear_annotations = get_clear_annotations(annotations, joined_sentence)
    connected_annotations = connect_annotations(clear_annotations, joined_sentence)
    filtered_annotations = filter_annotations(connected_annotations, tokenized_sentence)

    return filtered_annotations


def inference_wikidump(dataset_path, pipeline):
    '''Get predicted annotations for jsonl wikidump dataset'''
    n_documents = 99999
    wikidump_labeled = []

    num_lines = sum(1 for line in open(dataset_path))

    with jsonlines.open(dataset_path) as dataset_file:
        document_count = 0
        print(f'WIKIANN SLOVAK BERT INFERENCING {dataset_path}')
        for doc in tqdm(dataset_file, total=num_lines):

            ner_data = []
            tokenized_text = tokenize_nltk(doc['text'])
            for sentence in tokenized_text:
                annotations = inference(pipeline, sentence)
                ner_data.append({
                    'text': ' '.join(sentence),
                    'ents': annotations,
                })

            document_count += 1
            wikidump_labeled.append({
                'sentences': ner_data,
                'meta': doc['meta'],
            })

            if document_count >= n_documents:
                break
    #visualize_ner({'text': ' '.join(tokenized_text[0]), 'ents': annotations})
    #visualize_ner(ner_data, out_path='visualized_ner_wikidump.html')

    return wikidump_labeled


def find_token_index(span_position, tokens):
    '''Get token belonging to position in sentence'''
    for i, token in enumerate(tokens):
        if token['start'] <= span_position < token['end']:
            return i
    return -1


def wikidump_to_prodigy(wikidump_labeled, output_path):
    '''Join predictions in span format for separate sentences to
    predictions for the whole article which will load to prodigy.
    '''
    docs = []

    print('CONVERTING SLOVAK BERT WIKI ANNOTATIONS TO PRODIGY FORMAT')
    for doc in tqdm(wikidump_labeled):
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


def main():
    '''Predict and process NER tags on wikipedia articles
    and convert it to Prodigy format for manual annotation
    '''
    ner = pipeline(task='ner', model='./output/test-ner', aggregation_strategy="simple")

    dataset_path = '../wiki_dump/text_jsonl/20220316-170708.jsonl'
    output_path = '../../prodigy/wikiann_bert/20220316-170708_bert.jsonl'

    annotations = inference_wikidump(dataset_path, ner)
    wikidump_to_prodigy(annotations, output_path)

    #visualize_ner({'text': ' '.join(tokenized_text[0]), 'ents': annotations})


if __name__ == '__main__':
    main()
    #visualize_prodigy(annotations_prodigy)
