import pandas as pd

from logistic_regression import train_and_save as train_lr

from decision_tree import train_and_save as train_dt

from knn import train_and_save as train_knn

from naive_bayes import train_and_save as train_nb

from random_forest import train_and_save as train_rf

from xgboost_model import train_and_save as train_xgb


CSV_FILE = "ObesityDataSet.csv"

def main():
    df = pd.read_csv(CSV_FILE)

    train_lr(df, out_path="models/logistic_regression.pkl")
    train_dt(df, out_path="models/decision_tree.pkl")
    train_knn(df, out_path="models/knn.pkl")
    train_nb(df, out_path="models/naive_bayes.pkl")
    train_rf(df, out_path="models/random_forest.pkl")
    try:
        train_xgb(df, out_path="models/xgboost.pkl")
    except Exception as e:
        print(f"Skipped XGBoost: {e}")

    print("\nDone. All selected models trained and saved.")

if __name__ == "__main__":
    main()
