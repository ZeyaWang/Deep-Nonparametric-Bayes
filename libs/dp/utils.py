import numpy as np


def logdet(A):
	U = np.linalg.cholesky(A)
	return 2*np.sum(np.log(np.diag(U)))