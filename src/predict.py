import pickle
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler


def del_feature(data_train, data_test):
	data_train["oringin"] = "train"
	data_test["oringin"] = "test"
	data_all = pd.concat([data_train, data_test], axis=0, ignore_index=True)
	"""删除特征"V5","V9","V11","V17","V22","V28"，训练集和测试集分布不一致"""
	data_all.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)
	# print('drop after data_all.shape=',data_all.shape)
	# figure parameters
	data_train = data_all[data_all["oringin"] == "train"].drop("oringin", axis=1)
	# print('drop after data_train.shape=',data_train.shape)

	"""'V14', u'V21', u'V25', u'V26', u'V32', u'V33', u'V34'"""
	# Threshold for removing correlated variables
	threshold = 0.1
	# Absolute value correlation matrix
	corr_matrix = data_train.corr().abs()
	drop_col = corr_matrix[corr_matrix["target"] < threshold].index
	# print('drop_col=',drop_col)
	data_all.drop(drop_col, axis=1, inplace=True)
	# print('data_all.shape=',data_all.shape)
	return data_all


def scale_minmax(col):
	return (col - col.min()) / (col.max() - col.min())


def normal(data_all):
	"""归一化"""
	cols_numeric = list(data_all.columns)
	cols_numeric.remove("oringin")
	scale_cols = [col for col in cols_numeric if col != 'target']
	print('scale_cols=', scale_cols)
	data_all[scale_cols] = data_all[scale_cols].apply(scale_minmax, axis=0)
	return data_all


"""extract test data (without SalePrice)"""


def get_test_data(data_all):
	df_test = data_all[data_all["oringin"] == "test"].reset_index(drop=True)
	return df_test.drop(["oringin", "target"], axis=1)


if __name__ == '__main__':
	with open("zhengqi_train.txt")  as fr:
		data_train = pd.read_table(fr, sep="\t")

	with open("zhengqi_test.txt") as fr_test:
		data_test = pd.read_table(fr_test, sep="\t")
	data_all = del_feature(data_train, data_test)
	print('clear data_all.shape', data_all.shape)

	data_all = normal(data_all)

	X_test = get_test_data(data_all)
	print('X_test.shape', X_test.shape)
	poly_trans = PolynomialFeatures(degree=2)
	X_test = poly_trans.fit_transform(X_test)
	print(X_test.shape)

	with open("forest_model.pkl", "rb") as f:
		model = pickle.load(f)
		X_pre = model.predict(X_test)
		print(X_pre.shape)
		X_pre = list(map(lambda x: round(x, 3), X_pre))
		X_pre = np.reshape(X_pre, (-1, 1))
		print(X_pre.shape)
		X_pre = pd.DataFrame(X_pre)
		print(X_pre)
		X_pre.to_csv('result.txt', index=False, header=False)