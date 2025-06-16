RETRIEVALS_PORTERLUCENE = ["AxiomaticF1EXP", "BM25", "Classic", "DFIStandard", "InExpB2", "LMD", "LMJM"]
RETRIEVALS_PORTERLUCENE = [f"porter-lucene-{x}" for x in RETRIEVALS_PORTERLUCENE]

RETRIEVALS_RUN = ["bi-encoder", "cocondenser", "colbertv2", "contriever", "splade-cocondenser-ensemble-distil", "splade-max", "tasb"]
RETRIEVALS_RUN = [f"run-{x}" for x in RETRIEVALS_RUN]

PREDICTORS_BERT = ["neuralqpp", "qppbertpl", "deepqpp", "bertqpp"]
PREDICTORS_POST = ["Clarity", "NQC", "SMV", "WIG"]
PREDICTORS_POSTUEF = ["UEFClarity", "UEFNQC", "UEFSMV", "UEFWIG"]
PREDICTORS_PRE_ALL = ["ICTFavg", "ICTFmax", "ICTFstd", "ICTFsum",
                  "IDFavg", "IDFmax", "IDFstd", "IDFsum",
                  "SCQavg", "SCQmax", "SCQstd", "SCQsum",
                  "SCSsum", "VARavg", "VARmax", "VARstd", "VARsum"]
PREDICTORS_PRE = ["SCSsum", "SCQavg", "SCQmax", "ICTFavg", "ICTFmax", "IDFavg", "IDFmax", "VARavg", "VARmax"]

RETRIEVALS_ALL = [*RETRIEVALS_PORTERLUCENE]
PREDICTORS_ALL = [*PREDICTORS_PRE, *PREDICTORS_POST, *PREDICTORS_POSTUEF, *PREDICTORS_BERT]
BASE_PATH = ""