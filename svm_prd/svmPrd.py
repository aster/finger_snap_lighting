#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sklearn
from sklearn.svm import SVC
import numpy as np
import joblib
np.random.seed(0)
 
def main():
    # 1. reading data
    xtrain,ttrain=[],[]
    fin=open("../make_data_set/learning_data.txt","r")
    for i,line in enumerate(fin):
        line=line.rstrip()
        if line:
            tmp=line.split("\t")
            tmpx=tmp[0].split(",")
            tmpx=[float(j) for j in tmpx]
            tmpt=int(tmp[1])
            xtrain.append(tmpx)
            ttrain.append(tmpt)
    fin.close()
    xtrain=np.asarray(xtrain,dtype=np.float32)
    ttrain=np.asarray(ttrain,dtype=np.int32)
     
    # 2. learning, cross-validation
    diparameter={"kernel":["rbf"],"gamma":[10**i for i in range(-4,2)],"C":[10**i for i in range(-2,4)],"random_state":[123],}
    licv=sklearn.model_selection.GridSearchCV(SVC(),param_grid=diparameter,scoring="accuracy",cv=5,n_jobs=5)
    licv.fit(xtrain,ttrain)
    predictor=licv.best_estimator_
    joblib.dump(predictor,"predictor_svc.pkl",compress=True)
     
    # 3. evaluating the performance of the predictor
    liprediction=predictor.predict(xtrain)
    table=sklearn.metrics.confusion_matrix(ttrain,liprediction)
    tn,fp,fn,tp=table[0][0],table[0][1],table[1][0],table[1][1]
    print("TPR\t{0:.3f}".format(tp/(tp+fn)))
    print("SPC\t{0:.3f}".format(tn/(tn+fp)))
    print("PPV\t{0:.3f}".format(tp/(tp+fp)))
    print("ACC\t{0:.3f}".format((tp+tn)/(tp+fp+fn+tn)))
    print("MCC\t{0:.3f}".format((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)))
    print("F1\t{0:.3f}".format((2*tp)/(2*tp+fp+fn)))
     
    # 4. printing parameters of the predictor
    print(sorted(predictor.get_params(True).items()))
    print(predictor.support_vectors_)
    print(predictor.dual_coef_)
    print(predictor.intercept_)
    print(predictor.gamma)
     
if __name__ == '__main__':
    main()
