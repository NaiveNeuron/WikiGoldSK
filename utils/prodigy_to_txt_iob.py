import spacy
import json
import jsonlines
import os
import re


def clear_xml_tags(xml_path, output_path):
    '''Remove xml tags from dataset which is separated to documents'''
    with open(xml_path, 'r') as f:
        data = f.read()

    begin_tag = r"<doc id=\"[0-9]+\">\n\n"
    end_tag = r"</doc>\n\n"

    data = re.sub(begin_tag, "", data)
    data = re.sub(end_tag, "", data)

    with open(output_path, 'w') as f:
        f.write(data)


def fix_abbreviation_tokenization(txt_iob_path, same_file=True):
    '''Repair wrong nltk tokenization of abbreviations'''
    abbreviations = ['napr', 'lat', 'rod', 'mr', 'sv', 'mgr',
       'prof', 'ing', 'bc', 'dr', 'rus', 'tzv', 'phd', 'drsc', 'phdr',
       'dr', 'iii', 'ii', 'i', 'iv', 'odd', 'angl', 'skr', 'stor',
       'pol', 'vz', 'tal', 'rndr', 'fr', 'odd', 'mad', 'var', 'grec',
       'gr', 'nem', 'lat', 'hebr', 'arab', 'novoheb',
        'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
        'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'
    ]

    patterns = [
        {'what': r"\n({}) O\n\. (.*)\n", 'to': r'\n\1. O\n'},
        {'what': r"\n({}) (B-|I-)(PER|LOC|ORG|MISC)\n\. (.*)\n", 'to': r'\n\1. \2\3\n'},
    ]

    # Patterns to help locate sentences ended with abbreviation - has to be manual evaluated since sometimes it is right
    # patterns = [
    #     # r"\n(([^ \n])+|([^ \n]+ [^ \n]+)|([^ \n]+ [^ \n]+ [^ \n]+)\.)\n",
    #     {'what': r"\n({}) O\n\. (.*)\n\n", 'to': r'\n\1. O\nZZZ\n'},
    #     {'what': r"\n({}) (B-|I-)(PER|LOC|ORG|MISC)\n\. (.*)\n\n", 'to': r'\n\1. \2\3\nZZZ\n'},
    # ]

    with open(txt_iob_path, 'r') as f:
        data = f.read()

    for pattern in patterns:
        for abbr in abbreviations:
            what, to = pattern['what'], pattern['to']
            what = what.replace('{}', abbr)
            data = re.sub(what, to, data, re.M)

    suffix = ''
    if not same_file:
        suffix = '.cleared'

    with open(f'{txt_iob_path}{suffix}', 'w') as f:
        f.write(data)


def clear_txt_iob(txt_iob_path, same_file=True):
    '''Replace some common mistake patters'''
    patterns = [
        # r"\n(([^ \n])+|([^ \n]+ [^ \n]+)|([^ \n]+ [^ \n]+ [^ \n]+)\.)\n",
        {'what': r"\` O\n\` O", 'to': r'" O'},
        {'what': r"\` (B-|I-)(PER|LOC|ORG|MISC)\n\` (B-|I-)(PER|LOC|ORG|MISC)", 'to': r'" \1\2'},
        {'what': r"'' O", 'to': r'" O'},
        {'what': r"'' (B-|I-)(PER|LOC|ORG|MISC)", 'to': r'" \1\2'},
        {'what': r"' O", 'to': r'" O'},
        {'what': r"(“|„)", 'to': r'"'},
        {'what': r"([0-9]{4})([0-9]{4}) O\n", 'to': r'\1 O\n- O\n\2 O\n'},
        {'what': r"\n([0-9]{4})–([0-9]{4}) O\n", 'to': r'\n\1 O\n- O\n\2 O\n'},
        {'what': r"­", 'to': r''},
        {'what': r"([0-9]+) O\n\. O\n([^\n])", 'to': r'\1. O\n\2'},
        {'what': r"napr", 'to': r'\1. O\n\2'},
    ]

    with open(txt_iob_path, 'r') as f:
        data = f.read()

    for pattern in patterns:
        while re.search(pattern["what"], data, re.M):
            data = re.sub(pattern["what"], pattern["to"], data, re.M)

    suffix = ''
    if not same_file:
        suffix = '.cleared'

    with open(f'{txt_iob_path}{suffix}', 'w') as f:
        f.write(data)


def iob_to_txt_file(output_path, docs):
    '''Save IOB data to txt format'''
    #TODO 1 time in train dataset there are 4 newlines in row
    with open(output_path, 'w') as f:
        for doc in docs:
            tokens = doc['tokens']
            tags = doc['tags']
            #exclude docs without any entities
            if len([t for t in tags if t != 'O']) == 0:
                continue
            print(f'<doc id="{doc["id"]}">\n', file=f)
            for token, tag in zip(tokens, tags):
                if str(token) == '\n\t':
                    print('\n', end='', file=f)
                else:
                    print(token, tag, file=f)
            print(f'\n</doc>\n', file=f)


def get_iob(data):
    '''Get IOB tags from Prodigy/Spacy format'''
    docs = []
    # create example only if doc was accepted in prodigy
    examples = ((eg["text"], eg) for eg in data if eg['answer'] == 'accept')
    nlp = spacy.blank("en")
    for doc, eg in nlp.pipe(examples, as_tuples=True):
        doc.ents = [doc.char_span(s["start"], s["end"], s["label"]) for s in eg["spans"]]
        iob_tags = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ else "O" for t in doc]
        docs.append({
            'tokens': [t for t in doc],
            'tags': iob_tags,
            'id': eg['meta']['id'],
        })

    return docs


def convert_single_file(prodigy_path, txt_path):
    '''Convert Prodigy labeled dataset to IOB txt format'''
    with jsonlines.open(prodigy_path, 'r') as f:
        data = list(f)

    iob_to_txt_file(txt_path, get_iob(data))


def convert_all_files(prodigy_dir, txt_file):
    '''Convert Prodigy labeled dataset files from directory to single IOB txt format file'''
    all_docs = []
    for root, dirs, files in os.walk(prodigy_dir):
        for name in sorted(files):
            with jsonlines.open(os.path.join(prodigy_dir, name), 'r') as prodigy_file:
                data = list(prodigy_file)
                docs = get_iob(data)
                all_docs.extend(docs)
    iob_to_txt_file(txt_file, all_docs)


def main():
    # prodigy_path = 'prodigy-data/annotations_prod13.jsonl'
    # txt_path = 'prodigy-data/annotations_prod13.txt'
    # convert_single_file(prodigy_path, txt_path)

    prodigy_dir = '../prodigy_jsonl'
    iob_txt_file = '../final_txt/dataset.uncleared.txt'
    convert_all_files(prodigy_dir, iob_txt_file)

    #functions are meant to be used separately
    #clear_txt_iob(iob_txt_file, same_file=True)
    #fix_abbreviation_tokenization('iob_txt_file', same_file=True)
    #clear_xml_tags(iob_txt_file, iob_txt_file)


if __name__ == '__main__':
    main()
