import numpy as np


class DCP_XPd_PWL:
	'''Piece-wise Linear map for getting dcp and X_Pd'''
	def __init__(self, DCP, X_Pd):
		self.DCP = np.array(DCP)
		self.X_Pd = np.array(X_Pd)

	def get_XPd(self, dcp):
		a = np.where(self.DCP>dcp)[0][0]
		b = a - 1
		x_Pd = self.X_Pd[b] + (self.X_Pd[a] - self.X_Pd[b]) * (dcp - self.DCP[b])/(self.DCP[a]-self.DCP[b])
		return x_Pd

	def get_dcp(self, x_Pd):
		a = np.where(self.X_Pd>x_Pd)[0][0]
		b = a - 1
		x_Pd = self.DCP[b] + (self.DCP[a] - self.DCP[b]) * (x_Pd - self.X_Pd[b])/(self.X_Pd[a]-self.X_Pd[b])
		return dcp

