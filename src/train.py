import os
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get('TRAINING_DATA')
FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get('MODEL')

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(['id','target','kfold'],axis=1)
    valid_df = valid_df.drop(['id','target','kfold'],axis=1)

    valid_df = valid_df[train_df.columns] # the order of variables is the same

    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c,lbl))

    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df,ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(label_encoders,f"models/{MODEL}_label_encoder.pkl")
    joblib.dump(clf,f"models/{MODEL}.pkl")
