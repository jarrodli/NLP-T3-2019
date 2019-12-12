import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
from tika import parser
import numpy as np
import collections
import math
import sys
import os
import re

IDF_T = {}
STOPWORDS = []
TOTAL_DOCS = 0
TOTAL_TERMS = 0
K = 50

# !! IF THERE IS A READ ERROR DELETE .DS_STORE !!
# tf_idf calculations are based off of ::
# http://www.tfidf.com/

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
		global TOTAL_TERMS

		for k, v in self._frequency_dict.items():
			if k not in IDF_T:
				IDF_T[k] = 1
				TOTAL_TERMS += 1
			else:
				IDF_T[k] += 1

	def set_vector(self):
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
		return self._vec

def convert_files():

	c_path = str(os.getcwd())
	pdf_path = c_path + "/inspect_pdfs"
	directory = os.fsencode(pdf_path)
	i = 0
	for file in os.listdir(directory):
		fname = os.fsdecode(file)
		os.chdir(pdf_path)
		if fname.endswith(".pdf"):
			raw = parser.from_file(fname)
			os.chdir(c_path + "/orig_text")
			current_output = "%s.txt" % i
			with open(current_output, 'w+') as file:
				file.write(raw['content'])
				i += 1

	os.chdir(c_path)			

def plot(M, *args):

	x,y = M.T

	plt.scatter(x,y, c='blue')

	# plots a comparision document
	if len(args) > 0:
		x,y = args[0].T
		plt.scatter(x, y, c='red')
	
	plt.show()

def write_matrix(M):

	np.savetxt('test.out', M, delimiter=',', fmt='%1.8f')

def write_forder(fnames):
	with open("fnames.out", "w+") as file:
		for f in fnames:
			file.write(str(f) + "\n")

def import_stopwords():

	with open("stopword.txt", "r") as file:
		temp = file.read().split()
		for word in temp:
			STOPWORDS.append(word)

def create_tf_idfs():

	documents = []
	f_names = []
	# create tf_idf vectors
	dir_len = 0
	c_path = os.getcwd()

	os.chdir(c_path + "/orig_text")
	directory = os.fsencode(os.getcwd())
	for file in os.listdir(directory):
		f_names.append(str(file))
		fname = os.fsdecode(file)
		documents.append(tf_idf())
		documents[dir_len].calcluate_frequency(fname)
		dir_len += 1
	
	for i in range(dir_len):
		documents[i].calc_doc_idf()

	for i in range(dir_len):
		documents[i].set_vector()

	os.chdir(c_path)
	return documents, f_names

def create_np_matrix(tf_idfs):

	vec_list = []
	for i in range(TOTAL_DOCS):
		vec_list.append(tf_idfs[i].get_vector())

	# each index of vec_list currently contains a column vector
	# therefore we transpose the final matrix 
	return np.transpose(np.array(vec_list))

# lsa calculations are based off of:
# https://en.wikipedia.org/wiki/Latent_semantic_analysis

def latent_semantic_analysis(M):
	sys.stderr.write("calculating SVD...\n")
	# carry out the svd of the matrix M
	M = csc_matrix(M, dtype=float)
	U, s, VT  = svds(M,k=K)

	'''
	# turn the singular value list s into an n x m matrix
	sv_matrix = np.zeros((TOTAL_TERMS,TOTAL_DOCS))
	for i in range(min(TOTAL_TERMS,TOTAL_DOCS)):
		sv_matrix[i, i] = s[i]

	# rank reduce diagonal matrix
	sv_matrix = sv_matrix[0:K]
	sv_matrix = sv_matrix[:,0:K]

	# rank reduce inverse matrix
	VT = VT[0:K]

	# rank reduce eigenvector matrix
	U = U[:,0:K]
	'''
	V = np.transpose(VT)
	print(str(V))
	sys.stderr.write("finished calculating SVD\n")
	return V, U, s

def cosine_similarity(M, fnames):

	sys.stderr.write("calculating cosine cosine_similarity...\n")
	adj_M = np.zeros((TOTAL_DOCS+1, TOTAL_DOCS+1))
	#M = np.transpose(M)

	for i in range(len(M)):
		for j in range(i):
			v1_denom = math.sqrt(np.sum(np.square(M[i])))
			v2_denom = math.sqrt(np.sum(np.square(M[j])))
			adj_M[i+1,j+1] = math.acos(np.dot(M[i], M[j]) / (v1_denom * v2_denom))

	adj_M = np.around(np.reciprocal(adj_M, where=adj_M!=0), decimals=2)

	for i in range(len(M)):
		fnames[i] = re.sub("b'","",fnames[i])
		fnames[i] = re.sub(".txt'","",fnames[i])
		fnames[i] = int(fnames[i])

		adj_M[0,i+1] = fnames[i]
		adj_M[i+1,0] = fnames[i]

	adj_M[0][0] = 0.0 # sanity assignment

	print(str(adj_M))
	return adj_M
	

if __name__ == '__main__':

	convert_files()
	import_stopwords()
	
	tf_idfs, f_names = create_tf_idfs()
	M = create_np_matrix(tf_idfs)
	RR_M, U, s = latent_semantic_analysis(M)
	write_matrix(RR_M)
	c_dot = None
	
	if len(sys.argv) > 1:
		c_tfv = tf_idf()
		with open(str(sys.argv[1]), "r") as file:
			c_tfv.calcluate_frequency(str(sys.argv[1]))
			c_tfv.set_vector()

		q = np.array(c_tfv.get_vector())
		c_dot = np.dot(q, np.dot(U, s))
		# append the coordinates of the test file to f_names
		f_names.append(c_dot)

	write_forder(f_names)
	adj_M = cosine_similarity(RR_M, f_names)
	np.savetxt("data.csv", adj_M, delimiter=",", fmt='%0.2f')
	
	'''
	if c_dot is not None:
		plot(RR_M, c_dot)
	else:
		plot(RR_M)
	'''

