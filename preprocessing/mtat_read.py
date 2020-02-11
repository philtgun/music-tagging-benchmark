import os
import numpy as np
import glob
from essentia.standard import MonoLoader
import fire
import tqdm
import argparse
import csv


class Processor:
	def __init__(self, config):
		self.fs = 16000
		self.input = config.input
		self.output = config.output
		self.dataset = config.dataset

		if config.dataset == 'jamendo':
			self.keep_structure = 1

		elif config.dataset == 'msd':
			self.keep_structure = 3
		else:
			self.keep_structure = 0

	def get_jamendo_paths(self, tsv_file):
		with open(tsv_file) as fp:
			reader = csv.reader(fp, delimiter='\t')
			next(reader, None)  # skip header
			paths = [row[3] for row in reader]
		return paths

	def get_paths(self):
		if self.dataset == 'jamendo':
			paths = self.get_jamendo_paths('./../split/mtg-jamendo/autotagging_top50tags-test.tsv')
			self.files = [os.path.join(self.input, path) for path in paths]
		else:
			self.files = glob.glob(os.path.join(self.input, '*/*.mp3'))
		self.npy_path = os.path.join(self.output, 'npy')
		if not os.path.exists(self.npy_path):
			os.makedirs(self.npy_path)

	def get_npy(self, fn):
		loader = MonoLoader(filename=fn, sampleRate=self.fs)
		x = loader()
		return x

	def iterate(self):
		self.get_paths()
		for fn in tqdm.tqdm(self.files):
			fn_with_structure = '/'.join(fn.split('/')[-1-self.keep_structure:])[:-3] + 'npy'
			npy_fn = os.path.join(self.npy_path, fn_with_structure)
			if not os.path.exists(npy_fn):
				os.makedirs(os.path.dirname(npy_fn), exist_ok=True)
				try:
					x = self.get_npy(fn)
					np.save(open(npy_fn, 'wb'), x)
				except RuntimeError:
					# some audio files are broken
					print(fn)
					continue


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str)
	parser.add_argument('--output', type=str)
	parser.add_argument('--dataset', type=str, choices=['mtat', 'jamendo', 'msd'])
	config = parser.parse_args()

	p = Processor(config)
	# fire.Fire({'run': p.iterate})
	p.iterate()
