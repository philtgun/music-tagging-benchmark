import os
import numpy as np
import glob
from essentia.standard import MonoLoader
import fire
import tqdm
import argparse


class Processor:
	def __init__(self, config):
		self.fs = 16000
		self.input = config.input
		self.output = config.output
		self.keep_structure = config.keep_structure

	def get_paths(self):
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
			fn_with_structure = '/'.join(fn.split('/')[-2-self.keep_structure:-1])
			npy_fn = os.path.join(self.npy_path, fn_with_structure[:-3]+'npy')
			if not os.path.exists(npy_fn):
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
	parser.add_argument('--keep-structure', type=int, default=0)
	config = parser.parse_args()

	p = Processor(config)
	# fire.Fire({'run': p.iterate})
	p.iterate()
