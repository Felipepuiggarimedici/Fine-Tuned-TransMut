import pandas as pd
import numpy as np

def hyperparameterSelection(printMccAndAuprBestVals = False):
    df = pd.read_csv('hyperparametersMainModel.csv')
    
    bestMCC = -1
    bestAUPR = 0
    maxScore = -np.inf
    maxScoreMcc = -np.inf
    maxScoreAupr = -np.inf
    bestVals = (0,0,0)
    bestValsMcc = (0,0,0)
    bestValsAupr = (0,0,0)
    for index, row in df.iterrows():
        mcc = row["mcc"]
        aupr = row["aupr"]
        score = (mcc + aupr)/2
        if  score > maxScore:
            bestVals = (row["n_heads"], row["n_layers"], row["d_k"])
            maxScore = score
        if mcc > maxScoreMcc:
            maxScoreMcc = mcc
            bestValsMcc = (row["n_heads"], row["n_layers"], row["d_k"])
        if aupr > maxScoreAupr:
            maxScoreAupr = aupr
            bestValsAupr = (row["n_heads"], row["n_layers"], row["d_k"])
    print('The best hyperparameters are: n_heads: {}, n_layers = {}, d_k = {}'.format(bestVals[0], bestVals[1], bestVals[2]))
    if printMccAndAuprBestVals:
        print('The best values for MCC are: n_heads: {}, n_layers = {}, d_k = {}'.format(bestValsMcc[0], bestValsMcc[1], bestValsMcc[2]))
        print('The best values for AUPR are: n_heads: {}, n_layers = {}, d_k = {}'.format(bestValsAupr[0], bestValsAupr[1], bestValsAupr[2]))
    return int(bestVals[0]), int(bestVals[1]), int(bestVals[2])