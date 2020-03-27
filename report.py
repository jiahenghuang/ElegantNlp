#coding:utf-8
import os
import pandas as pd
from sklearn.metrics import classification_report

res_path = "za_result"
for f in os.listdir(res_path):
    print("- "*20, f)
    df = pd.read_csv(os.path.join(res_path, f), sep="\t", header=None, encoding="utf-8", names=("y", "pred", "p"))
    print("mean: ", df["y"].mean())
    print(classification_report(df["y"], df["pred"]))