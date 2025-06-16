import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata
from functools import reduce
from config import BASE_PATH, PREDICTORS_ALL
from util import (
    correlation, getGeoRisk, getSARE, getSMARE, getBootstrapSamples,
    getGeoRiskBootstrap, getURiskBootstrap, getTRiskBootstrap,
    getCorrelationBootstrap, getSMAREBootstrap
)
from risk import URisk, TRisk

def read_prediction_csv(dataset, retrieval, predictor, col_idx):
    return pd.read_csv(
        f"{BASE_PATH}/data/{dataset}/{dataset}_{retrieval}_{predictor}.csv",
        usecols=[0, col_idx],
        header=0,
        names=["tid", predictor]
    )

def merge_prediction_data(dataset, retrieval, predictors, col_idx):
    dfs = [read_prediction_csv(dataset, retrieval, p, col_idx) for p in predictors]
    merged = reduce(lambda left, right: pd.merge(left, right, on="tid", how="left"), dfs)
    return merged.fillna(0)

def getResults(retrieval, generate=True, willPrint=True, dataset="dplrnpass"):
    """
    Compute correlation and risk-based metrics for each predictor, save CSV if needed.
    """
    predictors = list(PREDICTORS_ALL)

    # Load data
    scores_df = merge_prediction_data(dataset, retrieval, predictors, col_idx=1)
    map_df = merge_prediction_data(dataset, retrieval, predictors, col_idx=2)

    # Traditional metrics
    metrics = {
        "corrKendall": [],
        "corrPearson": [],
        "corrSpearman": [],
        "smare": [],
    }

    for predictor in predictors:
        pred = scores_df[predictor].to_numpy()
        true = map_df[predictor].to_numpy()
        metrics["corrKendall"].append(correlation("kendall")(pred, true).statistic)
        metrics["corrPearson"].append(correlation("pearson")(pred, true).statistic)
        metrics["corrSpearman"].append(correlation("spearman")(pred, true).statistic)
        metrics["smare"].append(1 - getSMARE(pred, true))

    # Risk metrics
    sare_mat = np.array([getSARE(scores_df[p].to_numpy(), map_df[p].to_numpy()) for p in predictors]).T
    sare_inv = 1 - sare_mat
    sare_mean = np.mean(sare_inv, axis=1)

    risk_metrics = {
        "snGeoRiskInvA1": getGeoRisk(sare_inv, alpha=1),
        "snGeoRiskInvA5": getGeoRisk(sare_inv, alpha=5),
        "snGeoRiskInvA10": getGeoRisk(sare_inv, alpha=10),
        "snGeoRiskInvA20": getGeoRisk(sare_inv, alpha=20),
        "uRiskA1": [URisk(sare_inv[:, i], sare_mean, alpha=1) for i in range(sare_inv.shape[1])],
        "uRiskA5": [URisk(sare_inv[:, i], sare_mean, alpha=5) for i in range(sare_inv.shape[1])],
        "uRiskA10": [URisk(sare_inv[:, i], sare_mean, alpha=10) for i in range(sare_inv.shape[1])],
        "uRiskA20": [URisk(sare_inv[:, i], sare_mean, alpha=20) for i in range(sare_inv.shape[1])],
        "tRiskA1": [TRisk(sare_inv[:, i], sare_mean, alpha=1) for i in range(sare_inv.shape[1])],
        "tRiskA5": [TRisk(sare_inv[:, i], sare_mean, alpha=5) for i in range(sare_inv.shape[1])],
        "tRiskA10": [TRisk(sare_inv[:, i], sare_mean, alpha=10) for i in range(sare_inv.shape[1])],
        "tRiskA20": [TRisk(sare_inv[:, i], sare_mean, alpha=20) for i in range(sare_inv.shape[1])],
    }

    # Assemble final DataFrame
    df = pd.DataFrame({"predictor": predictors})
    for k, v in {**metrics, **risk_metrics}.items():
        df[k] = v

    # Sort and rank
    df.sort_values(by="corrKendall", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in df.columns[1:]:
        ranks = rankdata(-df[col].to_numpy(), method="min")
        df[col] = [f"{val:.3f} ({int(rk)})" for val, rk in zip(df[col], ranks)]

    if willPrint:
        print(df)

    if generate:
        df.to_csv(f"{BASE_PATH}/output/{dataset}_predictors-{retrieval}_data_riskqpp.csv", index=False)

def generateBootstrapCSVs(retrieval, n=1000, generate=True, willPrint=True, dataset="dplrnpass"):
    """
    Generate bootstrap-based samples of correlation and risk metrics.
    """
    predictors = list(PREDICTORS_ALL)

    true_df = merge_prediction_data(dataset, retrieval, predictors, col_idx=2).drop(columns="tid")
    scores_df = merge_prediction_data(dataset, retrieval, predictors, col_idx=1).drop(columns="tid")

    sare_mat = np.array([getSARE(scores_df[p].to_numpy(), true_df[p].to_numpy()) for p in predictors]).T
    sare_inv = 1 - sare_mat

    samples = getBootstrapSamples(n=n, n_queries=sare_inv.shape[0])

    # Bootstrap computations
    boot_metrics = {
        "snGeoRiskInvA1": getGeoRiskBootstrap(samples, sare_inv, alpha=1),
        "snGeoRiskInvA5": getGeoRiskBootstrap(samples, sare_inv, alpha=5),
        "snGeoRiskInvA10": getGeoRiskBootstrap(samples, sare_inv, alpha=10),
        "snGeoRiskInvA20": getGeoRiskBootstrap(samples, sare_inv, alpha=20),
        "uRiskA1": getURiskBootstrap(samples, sare_inv, alpha=1),
        "uRiskA5": getURiskBootstrap(samples, sare_inv, alpha=5),
        "uRiskA10": getURiskBootstrap(samples, sare_inv, alpha=10),
        "uRiskA20": getURiskBootstrap(samples, sare_inv, alpha=20),
        "tRiskA1": getTRiskBootstrap(samples, sare_inv, alpha=1),
        "tRiskA5": getTRiskBootstrap(samples, sare_inv, alpha=5),
        "tRiskA10": getTRiskBootstrap(samples, sare_inv, alpha=10),
        "tRiskA20": getTRiskBootstrap(samples, sare_inv, alpha=20),
        "corrPearson": getCorrelationBootstrap(samples, scores_df, true_df, "pearson"),
        "corrSpearman": getCorrelationBootstrap(samples, scores_df, true_df, "spearman"),
        "corrKendall": getCorrelationBootstrap(samples, scores_df, true_df, "kendall"),
        "smare": getSMAREBootstrap(samples, scores_df, true_df),
    }

    col_names = [str(i) for i in range(n)]
    for name, values in boot_metrics.items():
        df = pd.DataFrame(values.T, columns=col_names)
        df.insert(0, "predictor", predictors)
        if willPrint:
            print(df)
        if generate:
            df.to_csv(f"{BASE_PATH}/output/bootstrap/{dataset}_predictors-{retrieval}_{name}_bootstrap.csv", index=False)

def main():
    for dataset in ["dplrnpass", "robust04", "dplrnpass20"]:
        for retrieval in tqdm(["porter-lucene-BM25"]):
            getResults(retrieval, generate=True, willPrint=False, dataset=dataset)
            # Optional: uncomment to generate bootstrap
            # generateBootstrapCSVs(retrieval, generate=True, willPrint=False, dataset=dataset)

if __name__ == "__main__":
    main()