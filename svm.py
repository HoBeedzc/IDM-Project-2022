# add label to tf-idf dataframe,0 for baseball and 1 for hockey
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os


def add_label(df, label_dir):
    label = {}
    for dirname, _, filenames in os.walk(label_dir):
        for filename in filenames:
            label[filename] = dirname.split('/')[-1]
    for i in df.index:
        if label[i] == 'baseball':
            df.loc[i, 'LABEL'] = 0
        else:
            df.loc[i, 'LABEL'] = 1
    df["LABEL"] = df["LABEL"].astype("int")
    return df


# use Naive Bayes to classification the email.
def train_svm_model(data):
    models = []
    tests = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(data):
        gnb = SVC()
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        models.append(gnb.fit(train.iloc[:, :-1], train["LABEL"]))
        tests.append(test)
    return models, tests


# report result for the classification
# precision, recall, and F1-measure of each fold and the average values.


def report_result(models, tests):
    ps, rs, fs = [], [], []
    for i in range(len(models)):
        y_true = tests[i]["LABEL"]
        y_pred = models[i].predict(tests[i].iloc[:, :-1])
        ps.append(precision_score(y_true, y_pred))
        rs.append(recall_score(y_true, y_pred))
        fs.append(f1_score(y_true, y_pred))
        print(
            f"Fold {i}: precision {ps[-1]}, recall {rs[-1]}, F1-measure {fs[-1]}."
        )
    print(
        f"Avg: precision {np.mean(ps)}, recall {np.mean(rs)}, F1-measure {np.mean(fs)}."
    )


def main():
    email_dataset = add_label(tc_tfidf_df, BASE_DIR4)
    models, tests = train_svm_model(email_dataset)
    report_result(models, tests)


if __name__ == "__main__":
    main()