import trankit
from trankit.iterators.ner_iterators import NERDataset
import pickle
import tqdm


def save_trainer(trainer, save_path):
	with open(save_path, 'wb') as f:
		pickle.dump(trainer, f, pickle.HIGHEST_PROTOCOL)


def eval_trainer(trainer_path, test_dataset):
	with open(trainer_path, 'rb') as f:
		trainer = pickle.load(f)

	test_set = NERDataset(
		config=trainer._config,
		bio_fpath=test_dataset,
		evaluate=True
	)
	test_set.numberize()
	test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
	result = trainer._eval_ner(
			data_set=test_set,
			batch_num=test_batch_num,
			name='test',
			epoch=-1
	)
	print(result)


def train_test():

	trainer = trankit.TPipeline(
		training_config={
			'max_epoch': 10,
			'category': 'customized-ner',
			'task': 'ner', # task name
			'save_dir': './output_050222', # directory to save the trained model
			#'train_bio_fpath': '../data/train.txt',
			#'dev_bio_fpath': '../data/dev.txt',
			'train_bio_fpath': 'data/my_dataset/train.txt',
			'dev_bio_fpath': 'data/my_dataset/dev.txt',
			'max_input_length': 1000,
			'batch_size': 32,
		}
	)

	trainer._config.linear_dropout = 0.3
	trainer._config._cache_dir = './cache'

	print(', '.join("%s: %s" % item for item in vars(trainer._config).items()))


	trainer.train()

	test_set = NERDataset(
		config=trainer._config,
		#bio_fpath='../data/test.txt',
		bio_fpath='data/my_dataset/test.txt',
		evaluate=True
	)
	test_set.numberize()
	test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
	result = trainer._eval_ner(
			data_set=test_set,
			batch_num=test_batch_num,
			name='test',
			epoch=-1
		)
	print(result)
	#save_trainer(trainer, './output_dir/trainer.pkl')


def predict(gold_iob, out_path):

	p = trankit.Pipeline(lang='customized-ner', gpu=True, cache_dir='./output-article/gold')

	with open(gold_iob) as f:
		gold_data = f.read()
	sentences = gold_data.split('\n\n')

	with open(out_path, 'w') as f:
		for sentence in tqdm.tqdm(sentences):
			lines = sentence.split('\n')
			tokens = [l.split()[0] for l in lines]
			tags = [l.split()[1] for l in lines]
			predictions = p.ner(tokens, is_sent=True)
			for t in predictions['tokens']:
				if t['ner'] != 'O':
					pos, entity = t['ner'].split('-')
					if pos == 'S':
						pos = 'B'
					elif pos == 'E':
						pos = 'I'
					tag = f'{pos}-{entity}'
				else:
					tag = 'O'
				print(t['text'], tag, file=f)
			print('', file=f)

	#print(p.ner(['Donald', 'Trump', 'je', 'prvy', 'prezident', 'USA'], is_sent=True))



#train_test()
predict('./data/article/test-bsnlp.txt', './predictions-bsnlp.txt')