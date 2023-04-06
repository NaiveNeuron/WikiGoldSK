import jsonlines
import os
import re


def clear_xml_tags(xml_path, output_path):
    with open(xml_path, 'r') as f:
        data = f.read()

    begin_tag = r"<doc id=\"[0-9]+\">\n\n"
    end_tag = r"</doc>\n\n"

    data = re.sub(begin_tag, "", data)
    data = re.sub(end_tag, "", data)

    with open(output_path, 'w') as f:
        f.write(data)


def iob_to_conll_file(output_path, docs):
    #TODO 1 time in train dataset there are 4 newlines in row
    with open(output_path, 'w') as f:
        for doc in docs:
            tokens = doc['tokens']
            tags = doc['tags']
            #exclude docs without any entities
            # if len([t for t in tags if t != 'O']) == 0:
            #     continue
            print(f'<doc id="{doc["id"]}">\n', file=f)
            for token, tag in zip(tokens, tags):
                if str(token) == '\n\t':
                    print('\n', end='', file=f)
                else:
                    print(token, tag, file=f)
            print(f'\n</doc>\n', file=f)


def get_iob(data):
    docs = []
    for doc in data:
        tokens = [t['text'] for t in doc['tokens']]
        tags = ['O'] * len(tokens)

        for span in doc['spans']:
            l = span['label']
            t_s = span['token_start']
            t_e = span['token_end']

            tags[t_s] = f'B-{l}'
            x = t_s
            while x < t_e:
                x += 1
                tags[x] = f'I-{l}'
        docs.append({
            'tokens': tokens,
            'tags': tags,
            'id': doc['meta']['id'],
        })

    return docs


def convert_single_file(prodigy_path, conll_path):
    with jsonlines.open(prodigy_path, 'r') as f:
        data = list(f)

    iob_to_conll_file(conll_path, get_iob(data))


def convert_all_files(prodigy_dir, conll_file):
    all_docs = []
    for root, dirs, files in os.walk(prodigy_dir):
        for name in sorted(files):
            with jsonlines.open(os.path.join(prodigy_dir, name), 'r') as prodigy_file:
                data = list(prodigy_file)
                docs = get_iob(data)
                all_docs.extend(docs)
    iob_to_conll_file(conll_file, all_docs)


def main():
    prodigy_path = 'data/jozo/annotations-jozo-0-412.jsonl'
    conll_path = 'data/jozo/annotations-jozo-0-412.txt'
    #convert_single_file(prodigy_path, conll_path)

    # prodigy_dir = 'data'
    # conll_file = 'annotations_article01.txt'
    #convert_all_files(prodigy_dir, conll_file)

    clear_xml_tags(conll_path, 'data/jozo/annotations-jozo-full.txt')

if __name__ == '__main__':
    main()
