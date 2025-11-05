import yaml
import numpy as np
import pandas as pd

from utils import HTMLParser, ft_in_to_cm

# On suppose que HTMLParser et ft_in_to_cm existent déjà dans votre code
# from yourmodule import HTMLParser, ft_in_to_cm

def _load_config_and_codebook(config_path: str, codebook_html: str):
    parser = HTMLParser()
    codebook = parser.parse_html_codebook(codebook_html)
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config, codebook

def _classify_features(codebook, features):
    """Retourne un dict feature -> {'categorical'|'numerical'|'unknown'}"""
    features_classification = {}
    for feature in features:
        entry = next((item for item in codebook if item["sas_variable"] == feature), None)
        if entry:
            categories = sorted(set([cat['value'] for cat in entry['categories']]))
            if categories and categories[0] == str(1):
                features_classification[feature] = 'categorical'
            else:
                features_classification[feature] = 'numerical'
        else:
            features_classification[feature] = 'unknown'
    return features_classification

def _clean_and_engineer(X, features_classification):
    """Nettoyage et normalisation des variables brutes (sans scaling)."""
    # Colonnes continues
    continuous_columns = ['PHYSHLTH','MENTHLTH','POORHLTH',
                          'SLEPTIM1','WEIGHT2','HEIGHT3',
                          'MARIJAN1','ALCDAY4','AVEDRNK3','DRNK3GE5',
                          'MAXDRNKS','COPDSMOK','_PACKDAY']
    
    # Imputations/recodes spécifiques
    X["_PACKDAY"] = X["_PACKDAY"].fillna(X["_PACKDAY"].median())

    X["COPDSMOK"] = X["COPDSMOK"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["COPDSMOK"].median())
    X["MAXDRNKS"] = X["MAXDRNKS"].replace({77:np.nan, 99:np.nan}).fillna(X["MAXDRNKS"].median())
    X["DRNK3GE5"] = X["DRNK3GE5"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["DRNK3GE5"].median())
    X["AVEDRNK3"] = X["AVEDRNK3"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["AVEDRNK3"].median())
    X["MARIJAN1"] = X["MARIJAN1"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["MARIJAN1"].median())

    # ALCDAY4: semaines/mois -> jours
    mask_alc_week = X["ALCDAY4"].astype(float).between(101, 199)
    X.loc[mask_alc_week, "ALCDAY4"] = (X.loc[mask_alc_week, "ALCDAY4"] % 100) * 4
    mask_alc_month = X["ALCDAY4"].between(201, 299)
    X.loc[mask_alc_month, "ALCDAY4"] = (X.loc[mask_alc_month, "ALCDAY4"] % 200)
    X["ALCDAY4"] = X["ALCDAY4"].replace({888:0, 777:np.nan, 999:np.nan}).fillna(X["ALCDAY4"].median())

    X["SLEPTIM1"] = X["SLEPTIM1"].replace({77:np.nan, 99:np.nan}).fillna(X["SLEPTIM1"].median())
    X["POORHLTH"] = X["POORHLTH"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["POORHLTH"].median())
    X["MENTHLTH"] = X["MENTHLTH"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["MENTHLTH"].median())
    X["PHYSHLTH"] = X["PHYSHLTH"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["PHYSHLTH"].median())

    # WEIGHT2 conversion
    X["WEIGHT2"] = X["WEIGHT2"].astype(float)
    mask_lb = X["WEIGHT2"].between(50, 776)            # livres -> kg
    X.loc[mask_lb, "WEIGHT2"] = X.loc[mask_lb, "WEIGHT2"] * 0.45359237
    mask_kg = X["WEIGHT2"].between(9023, 9352)         # 9000 + kg
    X.loc[mask_kg, "WEIGHT2"] = X.loc[mask_kg, "WEIGHT2"] - 9000

    # HEIGHT3 conversion
    mask_ft_in = X["HEIGHT3"].astype(float).between(200, 711)  # piedpouces
    X.loc[mask_ft_in, "HEIGHT3"] = X.loc[mask_ft_in, "HEIGHT3"].apply(ft_in_to_cm)
    mask_cm = X["HEIGHT3"].between(9061, 9998)                 # 9000 + cm
    X.loc[mask_cm, "HEIGHT3"] = X.loc[mask_cm, "HEIGHT3"] - 9000

    X["WEIGHT2"] = X["WEIGHT2"].replace({7777:np.nan, 9999:np.nan}).fillna(X["WEIGHT2"].median())
    X["HEIGHT3"] = X["HEIGHT3"].replace({7777:np.nan, 9999:np.nan}).fillna(X["HEIGHT3"].median())

    # Catégorielles -> codes
    categorical_features = [f for f, kind in features_classification.items() if kind == 'categorical']
    if categorical_features:
        X[categorical_features] = X[categorical_features].fillna(-1)
        X[categorical_features] = X[categorical_features].astype('category').apply(lambda s: s.cat.codes)

    # Index
    X = X.set_index('ID')

    return X, continuous_columns