import jsonlines


def convert(txt_path, pet_path, n_labels=None, unlabeled=False):
    '''Created balanced dataset for PET experiments.

    param n_labels: Number of entities from each class in created dataset
    param unlabeled: Include tags in dataset

    IOB from txt format is converted to following jsonl format:
        {
            sentence: "<sentence>",
            tags: [<tags>],    # only if not unlabeled
        },

    From IOB tags are 'I-' and 'B-' removed.
    '''
    # dataset size limit in sentences
    N_SENTENCES = 1000
    jsonl = []
    labels_count = {'PER': 0, 'LOC': 0, 'ORG': 0, 'MISC': 0}

    with open(txt_path) as f:
        content = f.read()

    sentences = content.split('\n\n')[:N_SENTENCES]
    for sentence in sentences:
        words, tags, whole_tags = [], [], []
        for line in sentence.split('\n'):
            word, tag = line.split()
            whole_tags.append(tag)
            if tag != "O":
                tag = tag.split("-")[1]
            words.append(word)
            tags.append(tag)
        example = {
            'sentence': ' '.join(words)
        }
        if not unlabeled:
            example['tags'] = tags
            example['whole_tags'] = whole_tags

        if not unlabeled and n_labels is not None:

            # skip sententences without entities
            if len(example['tags']) == example['tags'].count('O'):
                continue

            # balance entities - include sentence in final dataset only if after addition
            # number of entities from each class won't exceed n_labels
            for label in labels_count.keys():
                count = example['whole_tags'].count(f'B-{label}')
                if labels_count[label] + count > n_labels:
                    break
            else:
                for label in labels_count.keys():
                    labels_count[label] += example['whole_tags'].count(f'B-{label}')

                example.pop("whole_tags", None)
                jsonl.append(example)
        else:
            example.pop("whole_tags", None)
            jsonl.append(example)

    with jsonlines.open(pet_path, mode='w') as writer:
        writer.write_all(jsonl)


def main():
    txt_path = '../../final_dataset/test.txt'
    pet_path = './data/test.jsonl'
    convert(txt_path, pet_path, None, unlabeled=False)


if __name__ == '__main__':
    main()
