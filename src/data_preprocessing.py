import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(path):
    df = pd.read_csv(path)

    df = df.rename(columns={"reamining_contract": "remaining_contract"})

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


def preprocess_data(df):
    X = df.drop("churn", axis=1)
    y = df["churn"]

    print("Feature columns used for training:", X.columns.tolist())
    print("Number of features:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, imputer