import pandas as pd

def clean_ogle_csv(input_path):
    df = pd.read_csv(input_path, dtype=str)
    # Remove spaces from column names
    df.columns = df.columns.str.replace(' ', '', regex=False)
    # Remove spaces from all string values using DataFrame.map (future-proof)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x.replace(' ', '') if isinstance(x, str) else x)
    df.to_csv(input_path, index=False)

if __name__ == '__main__':
    clean_ogle_csv('./data/classification_OGLE.csv')
