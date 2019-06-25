# coding:utf-8
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
# from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from beautifultable import BeautifulTable
import pickle
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier


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


"""function to get training samples"""


def get_training_data(data_all):
	df_train = data_all[data_all["oringin"] == "train"]
	y = df_train.target
	X = df_train.drop(["oringin", "target"], axis=1)
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=100)
	return X_train, X_valid, y_train, y_valid


"""extract test data (without SalePrice)"""


def get_test_data(data_all):
	df_test = data_all[data_all["oringin"] == "test"].reset_index(drop=True)
	return df_test.drop(["oringin", "target"], axis=1)


#
"""metric for evaluation"""


def rmse(y_true, y_pred):
	diff = y_pred - y_true
	sum_sq = sum(diff ** 2)
	n = len(y_pred)
	return np.sqrt(sum_sq / n)


def mse(y_ture, y_pred):
	return mean_squared_error(y_ture, y_pred)


"""function to detect outliers based on the predictions of a model"""


def find_outliers(model, X, y, sigma=3):
	# predict y values using model
	try:
		y_pred = pd.Series(model.predict(X), index=y.index)
	# if predicting fails, try fitting the model first
	except:
		model.fit(X, y)
		y_pred = pd.Series(model.predict(X), index=y.index)
	# calculate residuals between the model prediction and true y values
	resid = y - y_pred
	mean_resid = resid.mean()
	std_resid = resid.std()
	# calculate z statistic, define outliers to be where |z|>sigma
	z = (resid - mean_resid) / std_resid
	# 找出方差大于3的数据的索引，然后丢掉
	outliers = z[abs(z) > sigma].index
	# print and plot the results
	print('score=', model.score(X, y))
	print('rmse=', rmse(y, y_pred))
	print("mse=", mean_squared_error(y, y_pred))
	print('---------------------------------------')
	print('mean of residuals:', mean_resid)
	print('std of residuals:', std_resid)
	print('---------------------------------------')
	return outliers


def get_trainning_data_omitoutliers(X_t, y_t):
	y1 = y_t.copy()
	X1 = X_t.copy()
	return X1, y1


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


if __name__ == '__main__':
	with open("zhengqi_train.txt")  as fr:
		data_train = pd.read_table(fr, sep="\t")
	with open("zhengqi_test.txt") as fr_test:
		data_test = pd.read_table(fr_test, sep="\t")
	data_all = del_feature(data_train, data_test)
	print('clear data_all.shape', data_all.shape)
	data_all = normal(data_all)
	X_train, X_valid, y_train, y_valid = get_training_data(data_all)
	print('X_train.shape=', X_train.shape)
	print('X_valid.shape=', X_valid.shape)
	X_test = get_test_data(data_all)
	print('X_test.shape', X_test.shape)
	# find and remove outliers using a Ridge model
	outliers = find_outliers(Ridge(), X_train, y_train)
	""" permanently remove these outliers from the data"""
	X_train, y_train = get_trainning_data_omitoutliers(X_train.drop(outliers), y_train.drop(outliers))
	X1 = pd.concat([X_train, y_train], axis=1)
	X2 = pd.concat([X_valid, y_valid], axis=1)
	X_all = pd.concat([X1, X2], axis=0)
	print(X_all)
	y = X_all['target']
	X = X_all.drop(["target"], axis=1)
	print(X.shape)
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=100)
	poly_trans = PolynomialFeatures(degree=2)
	X_train = poly_trans.fit_transform(X_train)
	print(X_train.shape)
	X_valid = poly_trans.fit_transform(X_valid)
	print(X_valid.shape)
	print('==============forest_model========================')
	forest_model = RandomForestRegressor(
		n_estimators=500,
		criterion='mse',
		max_depth=20,
		min_samples_leaf=3,
		max_features=0.4,
		random_state=1,
		bootstrap=False,
		n_jobs=-1
	)
	forest_model.fit(X_train, y_train)
	importance = forest_model.feature_importances_
	table = BeautifulTable()
	# table.column_headers = ["feature", "importance"]
	print('RF feature importance:')
	# print(data_all)
	for i, cols in enumerate(X_all.iloc[:, :-1]):
		table.append_row([cols, round(importance[i], 3)])
	print(table)
	y_pred = forest_model.predict(X_valid)
	y_valid_rmse = rmse(y_valid, y_pred)
	print('y_valid_rmse=', y_valid_rmse)
	y_valid_mse = mse(y_valid, y_pred)
	print('y_valid_mse=', y_valid_mse)
	y_valid_score = forest_model.score(X_valid, y_valid)
	print('y_valid_score=', y_valid_score)
	with open("forest_model.pkl", "wb") as f:
		pickle.dump(forest_model, f)
	with open("forest_model.pkl", "rb") as f:
		model = pickle.load(f)
		y_pred = model.predict(X_valid)
		y_valid_rmse = rmse(y_valid, y_pred)
		print('y_valid_rmse=', y_valid_rmse)
		y_valid_mse = mse(y_valid, y_pred)
		print('y_valid_mse=', y_valid_mse)
		y_valid_score = model.score(X_valid, y_valid)
		print('y_valid_score=', y_valid_score)