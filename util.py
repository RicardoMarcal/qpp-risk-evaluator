import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata
from risk import geoRisk, URisk, TRisk
from sklearn.utils import resample

def correlation(type):
    if type == "spearman":
        return spearmanr
    elif type == "kendall":
        return kendalltau
    elif type == "pearson":
        return pearsonr
    else:
        raise ValueError("Invalid type")

def getGeoRisk(mat, alpha=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    geoRiskList = []
    for i, q in enumerate(mat[0]):
        gr = geoRisk(torch.tensor(mat), alpha, device, i=i, do_root=True).numpy()[0]
        geoRiskList.append(gr)
    
    return geoRiskList

def getSARE(predictor, baseline, method="average"):
    rx = rankdata(predictor, method=method)
    ry = rankdata(baseline, method=method)

    sARE = np.abs(rx - ry) / len(rx)
    
    return sARE

def getSMARE(predictor, baseline, method="average"):
    return np.mean(getSARE(predictor, baseline, method="average"))

def getBootstrapSamples(n_queries, n=1000):
    samples = []
    for i in range(n):
        sample = resample(range(n_queries), replace=True, random_state=i)
        samples.append(sample)
    return samples
        
def getGeoRiskBootstrap(samples, matrix, alpha=1):
    bootstrap_values = []

    # Realizando o bootstrapping
    for sample in samples:
        matrix_sample = matrix[sample, :]
        
        georisk = getGeoRisk(matrix_sample, alpha=alpha)
        bootstrap_values.append(georisk)
    
    return np.array(bootstrap_values)

def getURiskBootstrap(samples, matrix, alpha=1):
    bootstrap_values = []

    # Realizando o bootstrapping
    for sample in samples:
        matrix_sample = matrix[sample, :]
        
        sare_mean_baseline = np.mean(matrix_sample, axis=1)
        
        urisk = [URisk(matrix_sample[:, i], sare_mean_baseline, alpha) for i in range(matrix.shape[1])]

        bootstrap_values.append(urisk)
    
    return np.array(bootstrap_values)

def getTRiskBootstrap(samples, matrix, alpha=1):
    bootstrap_values = []

    # Realizando o bootstrapping
    for sample in samples:
        matrix_sample = matrix[sample, :]
        
        sare_mean_baseline = np.mean(matrix_sample, axis=1)
        
        trisk = [TRisk(matrix_sample[:, i], sare_mean_baseline, alpha) for i in range(matrix.shape[1])]
        
        bootstrap_values.append(trisk)
    
    return np.array(bootstrap_values)

def getCorrelationBootstrap(samples, scoresDF, ndcgDF, corrType):
    bootstrap_values = []

    # Realizando o bootstrapping
    for sample in samples:
        corr = [correlation(corrType)(scoresDF[predictor].to_numpy()[sample], ndcgDF[predictor].to_numpy()[sample]).statistic for predictor in scoresDF.columns]

        bootstrap_values.append(corr)
    
    return np.array(bootstrap_values)

def getSMAREBootstrap(samples, scoresDF, ndcgDF):
    bootstrap_values = []

    # Realizando o bootstrapping
    for sample in samples:
        smare = [1 - getSMARE(scoresDF[predictor].to_numpy()[sample], ndcgDF[predictor].to_numpy()[sample]) for predictor in scoresDF.columns]

        bootstrap_values.append(smare)
    
    return np.array(bootstrap_values)
