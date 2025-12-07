"""
@ IOC - CE IABD
"""
import unittest
import os
import pickle

from generardataset import generar_dataset
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

class TestGenerarDataset(unittest.TestCase):
	"""
	classe TestGenerarDataset
	"""
	global mu_p_be
	global mu_p_me
	global mu_b_bb
	global mu_b_mb
	global sigma
	global dicc

	mu_p_be = 3240 # mitjana temps pujada bons escaladors
	mu_p_me = 4268 # mitjana temps pujada mals escaladors
	mu_b_bb = 1440 # mitjana temps baixada bons baixadors
	mu_b_mb = 2160 # mitjana temps baixada mals baixadors
	sigma = 240 # 240 s = 4 min

	dicc = [
		{"name":"BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
		{"name":"MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
	]

	def test_longituddataset(self):
		"""
		Comprova que la longitud del dataset generat és correcta (25 per tipus, 100 total)
		"""
		arr = generar_dataset(25, 1, dicc)
		self.assertEqual(len(arr), 100)

	def test_valorsmitjatp(self):
		"""
		Comprova que la mitjana de tp per BEBB és propera a la mitjana esperada
		"""
		arr = generar_dataset(100, 1, [d for d in dicc if d['name'] == 'BEBB'])
		arr_tp = [row[1] for row in arr]
		tp_mig = sum(arr_tp) / len(arr_tp)
		self.assertTrue(abs(tp_mig - mu_p_be) < 200)

	def test_valorsmitjatb(self):
		"""
		Comprova que la mitjana de tb per MEMB és propera a la mitjana esperada
		"""
		arr = generar_dataset(100, 1, [d for d in dicc if d['name'] == 'MEMB'])
		arr_tb = [row[2] for row in arr]
		tb_mig = sum(arr_tb) / len(arr_tb)
		self.assertTrue(abs(tb_mig - mu_b_mb) < 200)

class TestClustersCiclistes(unittest.TestCase):
	"""
	classe TestClustersCiclistes
	"""
	@classmethod
	def setUpClass(cls):
		cls.path_dataset = './data/ciclistes.csv'
		cls.ciclistes_data = load_dataset(cls.path_dataset)
		cls.ciclistes_data_clean = clean(cls.ciclistes_data)
		cls.true_labels, cls.df_features = extract_true_labels(cls.ciclistes_data_clean)
		cls.clustering_model = clustering_kmeans(cls.df_features, n_clusters=4)
		with open('model/clustering_model.pkl', 'wb') as f:
			pickle.dump(cls.clustering_model, f)
		cls.data_labels = cls.clustering_model.labels_

	def test_check_column(self):
		"""
		Comprova que la columna 'tp' existeix al dataset
		"""
		self.assertIn('tp', self.df_features.columns)

	def test_data_labels(self):
		"""
		Comprova que el nombre de labels coincideix amb el nombre de files
		"""
		self.assertEqual(len(self.data_labels), len(self.df_features))

	def test_model_saved(self):
		"""
		Comprova que el fitxer clustering_model.pkl existeix a la carpeta model/
		"""
		check_file = os.path.isfile('./model/clustering_model.pkl')
		self.assertTrue(check_file)

if __name__ == '__main__':
	unittest.main()
