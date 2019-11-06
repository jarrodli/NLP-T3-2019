from tika import parser
import collections
import numpy as np
import math
import sys
import re

IDF_T = {}
NUM_DOCS = 0
STOPWORDS = []
TOTAL_DOCS = 0

class tf_idf:

	def __init__(self):
		global TOTAL_DOCS
		TOTAL_DOCS += 1
		self._frequency_dict = {}
		self._vec = []
		self._terms = 0

	def calcluate_frequency(self, finput):
		data = []
		skiplist = [",", ".", ";", "\"", "'", ":", "\\", \
					"/", "?", "-", "_", " ", ""]

		with open(finput, "r") as file:
			temp = file.read().split()
			for word in temp:
				if word in skiplist or word in STOPWORDS or \
				any(ch.isdigit() for ch in word):
					continue
				data.append(word.lower())
				self._terms += 1

		word_frequency = collections.Counter(data)
		
		self._frequency_dict = dict(word_frequency)

	def calc_doc_idf(self):
		global IDF_T

		for k, v in self._frequency_dict.items():
			if k not in IDF_T:
				IDF_T[k] = 1
			else:
				IDF_T[k] += 1

	def set_vector(self):
		global IDF_T

		for term in IDF_T:
			if term in self._frequency_dict:
				global TOTAL_DOCS
				tf = self._frequency_dict[term] / self._terms
				idf = math.log(TOTAL_DOCS / IDF_T[term])
				if idf <= 0:
					w_value = tf
				else:
					w_value = tf / idf
			else:
				w_value = 0.0

			assert w_value >= 0, "value, term " + str(w_value) + " " + term

			self._vec.append(w_value)

	def get_vector(self):
		return np.array(self._vec)


def convert_file():
	raw = parser.from_file('test4.pdf')

	with open("test4.txt", 'w') as file:
		file.write(raw['content'])

if __name__ == '__main__':
	#convert_file()

	with open("stopword.txt", "r") as file:
		temp = file.read().split()
		for word in temp:
			STOPWORDS.append(word)

	document1 = tf_idf()
	document2 = tf_idf()
	document1.calcluate_frequency("test2.txt")
	document2.calcluate_frequency("test4.txt")
	document1.calc_doc_idf()
	document2.calc_doc_idf()

	document1.set_vector()
	document2.set_vector()
	
	v1 = document1.get_vector()
	v1_denom = math.sqrt(np.sum(np.square(v1)))

	v2 = document2.get_vector()
	v2_denom = math.sqrt(np.sum(np.square(v2)))

	final = math.acos(np.dot(v1, v2) / (v1_denom * v2_denom))

	print(final)

