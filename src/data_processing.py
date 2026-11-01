import pandas as pd

def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    required_columns = ["Jam Tatap Muka", "Jam Online", "Nilai Akhir"]
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        return False, missing
    return True, []

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    initial_rows = df.shape[0]
    
    df_clean = df.drop_duplicates()
    after_duplicates = df_clean.shape[0]
    
    df_clean = df_clean.dropna(subset=["Jam Tatap Muka", "Jam Online", "Nilai Akhir"])
    
    for col in ["Jam Tatap Muka", "Jam Online", "Nilai Akhir"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        
    df_clean = df_clean.dropna()
    final_rows = df_clean.shape[0]
    
    stats = {
        "duplicates_removed": initial_rows - after_duplicates,
        "missing_removed": after_duplicates - final_rows
    }
    
    return df_clean, stats

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Jam Tatap Muka", "Jam Online", "Nilai Akhir"]].corr()