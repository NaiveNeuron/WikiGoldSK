import os
import random
import json
import jsonlines
from bs4 import BeautifulSoup as BSHTML
import time
import re

import nltk
nltk.download('punkt')


def is_doc_valid(doc_text):
    '''Limit length of wikipedia article'''
    MIN_LENGTH = 500  # average length of word is 4.7
    MAX_LENGTH = 5000

    l = len(doc_text)
    if l < MIN_LENGTH or l > MAX_LENGTH:
        return False

    return True


def remove_titles(doc):
    # main title
    header_pos = doc.find('\n')
    header = doc[:header_pos+1]
    title_pos = doc.find('\n\n')
    doc = header + doc[title_pos+1:]

    # other
    # replace sentences with 1-3 words bordered with newlines
    pattern = r"\n(([^ \n])+|([^ \n]+ [^ \n]+)|([^ \n]+ [^ \n]+ [^ \n]+)\.)\n"
    while re.search(pattern, doc, re.M):
        doc = re.sub(pattern, "\n", doc, re.M)

    return doc


def get_structured_docs(in_path, out_path):
    '''Parse cleaned wikipedia articles from XML and retrieve valid ones
    with ids for furhter usage
    '''
    ids = []
    d = 0
    for root, dirs, files in os.walk(in_path):
        for name in sorted(files):
            with open(os.path.join(in_path, name)) as in_f:
                content = in_f.read()

            BS = BSHTML(content)
            docs_in_file = BS.findAll("doc")
            valid_docs = [remove_titles(str(doc)) for doc in docs_in_file if is_doc_valid(doc.getText())]
            ids.extend([doc['id'] for doc in docs_in_file if is_doc_valid(doc.getText())])
            d += len(valid_docs)

            with open(os.path.join(out_path, name), 'w') as out_f:
                for doc in valid_docs:
                    print(doc, file=out_f)

            #break
        #break

    with open(os.path.join(out_path, 'ids'), 'w') as f:
        json.dump({'ids': ids}, f)

    print(d)
    return ids


def pick_random_docs(in_path, out_path, n_to_pick):
    '''Sample from valid articles that were not already picked'''
    timestr = time.strftime("%Y%m%d-%H%M%S")
    already_picked_ids_path = os.path.join(out_path, 'ids')
    already_picked_ids = set()
    already_picked = {}

    with open(os.path.join(in_path, 'ids')) as f:
        all_ids = set(json.load(f)['ids'])

    if os.path.exists(already_picked_ids_path):
        with open(already_picked_ids_path) as f:
            already_picked = json.load(f)
            already_picked_ids = set([id for id_field in already_picked.keys() for id in already_picked[id_field]])

    unpicked_ids = all_ids - already_picked_ids
    new_picked_ids = random.sample(unpicked_ids, n_to_pick)
    new_picked_docs = []

    for root, dirs, files in os.walk(in_path):
        for name in sorted(files):
            with open(os.path.join(in_path, name)) as in_f:
                content = in_f.read()

            BS = BSHTML(content)
            docs_in_file = BS.findAll("doc")
            new_picked_docs.extend([doc for doc in docs_in_file if doc['id'] in new_picked_ids])

            #break
        break

    with open(os.path.join(out_path, timestr), 'w') as out_f:
        for doc in new_picked_docs:
            print(doc, file=out_f)

    with open(already_picked_ids_path, 'w') as f:
        already_picked[timestr] = new_picked_ids
        json.dump(already_picked, f)


def save_docs(out_path, docs):
    with open(out_path, 'w', encoding="utf-8") as f:
        json.dump(docs, f)


def load_docs(path):
    with open(path) as f:
        return json.load(f)


def save_to_jsonl(in_path, out_path):
    '''Convert XML format of articles to jsonl'''
    file_name = in_path.split('/')[-1] + '.jsonl'
    out_path = os.path.join(out_path, file_name)

    docs = {}

    with open(in_path) as in_f:
        content = in_f.read()

    BS = BSHTML(content)
    docs_in_file = BS.findAll("doc")
    docs = [{'text': d.getText(), 'meta': {'id': d['id']}} for d in docs_in_file]

    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(docs)


def additional_remove_titles(in_path, out_path):
    '''Same functionality as remove_titles() on already sampled and saved articles'''
    with open(in_path) as in_f:
        content = in_f.read()

    BS = BSHTML(content)
    docs_in_file = BS.findAll("doc")
    docs = [remove_titles(str(doc)) for doc in docs_in_file]

    with open(out_path, 'w') as out_f:
        for doc in docs:
            print(doc, file=out_f)


def main():
    text_extracted_path = 'text_1/AA'
    text_valid = 'text_valid'
    text_picked = 'text_picked'
    n_docs = 50

    #ids = get_structured_docs(text_extracted_path, text_valid)                 # STEP 1: get valid docs - long enough from parsed dataset (already new enough)

    #picked_docs = pick_random_docs(text_valid, text_picked, n_docs)            # STEP 2: sampling from valid docs
    #additional_remove_titles('text_picked/20220316-170708', 'text_picked/20220316-170708')   # STEP 2b: remove titles (only when not parsing from already clear docs)

                                                                                # STEP 3: manually clear sampled docs

    save_to_jsonl('text_picked/20220316-170708', 'text_jsonl')                  # STEP 4: create jsonl for prodigy


if __name__ == '__main__':
    main()

