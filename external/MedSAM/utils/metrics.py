# -*- coding: utf-8 -*-
"""
Metrics for polyp segmentation evaluation (PraNet, MICCAI 2020)
Implements: Dice, IoU, F-measure, S-measure, Weighted F-measure, E-measure, Precision, Recall, Sensitivity, Specificity, MAE
"""

import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d

def dice_coefficient(pred, gt):
	pred = pred.astype(bool)
	gt = gt.astype(bool)
	inter = np.logical_and(pred, gt).sum()
	union = pred.sum() + gt.sum()
	if union == 0:
		return np.nan
	return 2 * inter / union

def iou_score(pred, gt):
	pred = pred.astype(bool)
	gt = gt.astype(bool)
	inter = np.logical_and(pred, gt).sum()
	union = np.logical_or(pred, gt).sum()
	if union == 0:
		return np.nan
	return inter / union

def precision_recall_specificity(pred, gt):
	pred = pred.astype(bool)
	gt = gt.astype(bool)
	TP = np.logical_and(pred, gt).sum()
	FP = np.logical_and(pred, ~gt).sum()
	TN = np.logical_and(~pred, ~gt).sum()
	FN = np.logical_and(~pred, gt).sum()
	precision = TP / (TP + FP + 1e-8)
	recall = TP / (TP + FN + 1e-8)
	specificity = TN / (TN + FP + 1e-8)
	sensitivity = recall
	return precision, recall, specificity, sensitivity

def f_measure(pred, gt):
	precision, recall, _, _ = precision_recall_specificity(pred, gt)
	if precision + recall == 0:
		return 0.0
	return 2 * precision * recall / (precision + recall)

def mae(pred, gt):
	pred = pred.astype(np.float64)
	gt = gt.astype(np.float64)
	return np.mean(np.abs(pred - gt))

def weighted_f_measure(pred, gt):
	# Implementation of original_WFb.m
	pred = pred.astype(np.float64)
	gt = gt.astype(bool)
	dGT = gt.astype(np.float64)
	E = np.abs(pred - dGT)
	Dst, IDXT = scipy.ndimage.distance_transform_edt(1 - dGT, return_indices=True)
	K = np.exp(-((np.arange(-3, 4)) ** 2) / (2 * 5 ** 2))
	K = np.outer(K, K)
	K = K / K.sum()
	Et = E.copy()
	Et[~gt] = E[IDXT[0][~gt], IDXT[1][~gt]]
	EA = convolve2d(Et, K, mode='same', boundary='symm')
	MIN_E_EA = E.copy()
	mask = gt & (EA < E)
	MIN_E_EA[mask] = EA[mask]
	B = np.ones(gt.shape)
	B[~gt] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5.0 * Dst[~gt])
	Ew = MIN_E_EA * B
	TPw = dGT.sum() - Ew[gt].sum()
	FPw = Ew[~gt].sum()
	R = 1 - Ew[gt].mean()
	P = TPw / (TPw + FPw + 1e-8)
	Q = 2 * R * P / (R + P + 1e-8)
	return Q

def enhanced_alignment_measure(pred, gt):
	# E-measure (Enhancedmeasure.m)
	pred = pred.astype(np.float64)
	gt = gt.astype(np.float64)
	if gt.sum() == 0:
		enhanced_matrix = 1.0 - pred
	elif np.all(gt == 1):
		enhanced_matrix = pred
	else:
		mu_pred = pred.mean()
		mu_gt = gt.mean()
		align_pred = pred - mu_pred
		align_gt = gt - mu_gt
		align_matrix = 2 * (align_gt * align_pred) / (align_gt ** 2 + align_pred ** 2 + 1e-8)
		enhanced_matrix = ((align_matrix + 1) ** 2) / 4
	return enhanced_matrix.mean()

def s_object(pred, gt):
	# S_object.m
	pred_fg = pred.copy()
	pred_fg[~gt] = 0
	O_FG = _object_score(pred_fg, gt)
	pred_bg = 1.0 - pred
	pred_bg[gt] = 0
	O_BG = _object_score(pred_bg, ~gt)
	u = gt.mean()
	return u * O_FG + (1 - u) * O_BG

def _object_score(pred, gt):
	if pred.size == 0 or not np.any(gt):
		return 0.0
	x = pred[gt].mean() if np.any(gt) else 0.0
	sigma_x = pred[gt].std() if np.any(gt) else 0.0
	return 2.0 * x / (x ** 2 + 1.0 + sigma_x + 1e-8)

def s_region(pred, gt):
	# S_region.m
	X, Y = _centroid(gt)
	GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
	pred_1, pred_2, pred_3, pred_4 = _divideprediction(pred, X, Y)
	Q1 = _ssim(pred_1, GT_1)
	Q2 = _ssim(pred_2, GT_2)
	Q3 = _ssim(pred_3, GT_3)
	Q4 = _ssim(pred_4, GT_4)
	return w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

def _centroid(gt):
	rows, cols = gt.shape
	if gt.sum() == 0:
		X = cols // 2
		Y = rows // 2
	else:
		total = gt.sum()
		i = np.arange(cols)
		j = np.arange(rows)
		X = int(np.round((gt.sum(axis=0) * i).sum() / total))
		Y = int(np.round((gt.sum(axis=1) * j).sum() / total))
	return X, Y

def _divideGT(gt, X, Y):
	hei, wid = gt.shape
	area = wid * hei
	LT = gt[:Y, :X]
	RT = gt[:Y, X:]
	LB = gt[Y:, :X]
	RB = gt[Y:, X:]
	w1 = (X * Y) / area
	w2 = ((wid - X) * Y) / area
	w3 = (X * (hei - Y)) / area
	w4 = 1.0 - w1 - w2 - w3
	return LT, RT, LB, RB, w1, w2, w3, w4

def _divideprediction(pred, X, Y):
	hei, wid = pred.shape
	LT = pred[:Y, :X]
	RT = pred[:Y, X:]
	LB = pred[Y:, :X]
	RB = pred[Y:, X:]
	return LT, RT, LB, RB

def _ssim(pred, gt):
	dGT = gt.astype(np.float64)
	N = pred.size
	x = pred.mean()
	y = dGT.mean()
	sigma_x2 = ((pred - x) ** 2).sum() / (N - 1 + 1e-8)
	sigma_y2 = ((dGT - y) ** 2).sum() / (N - 1 + 1e-8)
	sigma_xy = ((pred - x) * (dGT - y)).sum() / (N - 1 + 1e-8)
	alpha = 4 * x * y * sigma_xy
	beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)
	if alpha != 0:
		Q = alpha / (beta + 1e-8)
	elif alpha == 0 and beta == 0:
		Q = 1.0
	else:
		Q = 0.0
	return Q

def structure_measure(pred, gt):
	# StructureMeasure.m
	pred = pred.astype(np.float64)
	gt = gt.astype(bool)
	y = gt.mean()
	if y == 0:
		Q = 1.0 - pred.mean()
	elif y == 1:
		Q = pred.mean()
	else:
		alpha = 0.5
		Q = alpha * s_object(pred, gt) + (1 - alpha) * s_region(pred, gt)
		if Q < 0:
			Q = 0.0
	return Q
