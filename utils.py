import re, json, html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif


class HTMLParser():
    def __init__(self):
        pass

    def strip_tags(self, s: str) -> str:
        s = re.sub(r'<br\s*/?>', '\n', s, flags=re.I)
        s = re.sub(r'<[^>]+>', '', s)
        s = html.unescape(s)
        # normalize spaces
        s = re.sub(r'\xa0', ' ', s)  # non-breaking
        s = re.sub(r'[ \t]+', ' ', s).strip()
        return s

    def to_number(self, x):
        x = x.strip()
        if x in {'.', '. .', ''}:
            return None
        # remove commas in thousands
        x = x.replace(',', '')
        try:
            if '.' in x:
                return float(x)
            return int(x) if x.isdigit() else x
        except ValueError:
            return x

    def parse_html_codebook(self,path: str):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            snippet = f.read()

        table_blocks = re.findall(r'(<table class="table".*?</table>)', snippet, flags=re.S|re.I)


        items = []
        for tb in table_blocks:
            # header info
            m = re.search(r'<td[^>]*class="[^"]*linecontent[^"]*"[^>]*colspan="5"[^>]*>(.*?)</td>', tb, flags=re.S|re.I)
            meta = {}
            if m:
                meta_text = self.strip_tags(m.group(1))
                # Split on newlines and parse key: value
                for line in meta_text.split('\n'):
                    if ':' in line:
                        k, v = line.split(':', 1)
                        meta[k.strip()] = v.strip()
            # body rows
            body = re.search(r'<tbody>(.*?)</tbody>', tb, flags=re.S|re.I)
            rows = []
            if body:
                for tr in re.findall(r'<tr>(.*?)</tr>', body.group(1), flags=re.S|re.I):
                    tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, flags=re.S|re.I)
                    if len(tds) == 5:
                        value = self.strip_tags(tds[0])
                        vlabel = self.strip_tags(tds[1])
                        freq = self.strip_tags(tds[2])
                        perc = self.strip_tags(tds[3])
                        wperc = self.strip_tags(tds[4])
                        rows.append({
                            "value": value,
                            "label": vlabel,
                            "frequency": self.to_number(freq),
                            "percentage": self.to_number(perc),
                            "weighted_percentage": self.to_number(wperc),
                        })
            if meta or rows:
                items.append({
                    "sas_variable": meta.get("SAS Variable Name"),
                    "label": meta.get("Label"),
                    "section": meta.get("Section Name"),
                    "module_number": meta.get("Module Number"),
                    "question_number": meta.get("Question Number"),
                    "column": meta.get("Column"),
                    "type": meta.get("Type of Variable"),
                    "question": meta.get("Question"),
                    "raw_meta": meta,
                    "categories": rows
                })

        # Save to JSON file for download
        json_path = "data/codebook_parsed.json"
        with open(json_path, "w", encoding="utf-8") as f:
            dict = json.dump(items, f, ensure_ascii=False, indent=2)

        # Build a convenient DataFrame for querying: one row per (variable, value)
        records = []
        for item in items:
            for c in item["categories"]:
                records.append({
                    "sas_variable": item["sas_variable"],
                    "variable_label": item["label"],
                    "value": c["value"],
                    "value_label": c["label"],
                    "frequency": c["frequency"],
                    "percentage": c["percentage"],
                    "weighted_percentage": c["weighted_percentage"],
                })

        return items

    def generate_kept_feature_explainer(features: list, dict_data: list, output_path: str = "kept_features_explainer.txt"):
        """
        Génère un fichier texte décrivant chaque variable retenue.

        Args:
            features (list): Liste des variables retenues.
            dict_data (list): Dictionnaire contenant les métadonnées (sas_variable, label, question, categories...).
            output_path (str): Chemin du fichier texte de sortie.
        """

        with open(output_path, "w", encoding="utf-8") as f:
            for col in features:
                f.write(f"Variable: {col}\n")
                entry = next((item for item in dict_data if item.get("sas_variable") == col), None)

                if entry:
                    f.write(f"Label: {entry.get('label', 'N/A')}\n")
                    f.write(f"Question: {entry.get('question', 'N/A')}\n")
                    f.write("Categories:\n")

                    total = 0
                    for cat in entry.get("categories", []):
                        value = cat.get("value", "N/A")
                        label = cat.get("label", "N/A")
                        freq = cat.get("frequency", "N/A")
                        perc = cat.get("percentage", "N/A")

                        f.write(f"  - Value: {value}, Label: {label}, Freq: {freq}, Perc: {perc}\n")
                        if isinstance(freq, (int, float)):
                            total += freq

                    f.write(f"  Total Frequency: {total}\n")
                else:
                    f.write("No dictionary entry found.\n")

                f.write("\n" + "-"*60 + "\n\n")

        print(f"✅ Fichier généré : {output_path}")

def ft_in_to_cm(s):
    s = str(int(float(s))).zfill(3)  
    feet = int(s[0])
    inches = int(s[1:])
    return feet * 30.48 + inches * 2.54

def feature_classification_from_codebook(codebook: list, features: list) -> dict:
    classification = {}
    for item in codebook:
        if item.get("sas_variable") in features:
            var_name = item.get("sas_variable")
            
            categories = item.get("categories", [])
            print(categories)
            categories.sort(key=lambda x: x.get("value", 0))


            if categories and categories[0].get("value") != '1':
                classification[var_name] = "numerical"

            else:
                classification[var_name] = "categorical"
    return classification


# Handle missing values for categorical features
# categorical_features = [feat for feat, info in features_classification.items() if info['type'] == 'categorical']
# X[categorical_features] = X[categorical_features].fillna(-1)

# X[categorical_features] = X[categorical_features].astype('category')


def data_processing_train(df) :

    parser = HTMLParser()
    html_snippet = "data/USCODE22_LLCP_102523.HTML"

    codebook = parser.parse_html_codebook(html_snippet)

    config = "features.yaml"

    with open(config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    

    X = df.drop(columns='TARGET')    
    y = df['TARGET'].astype(int)


    X = X[config['features']].copy()

    features_classification = {}

    for feature in config['features']:
        features_classification[feature] = {}
        entry = next((item for item in codebook if item["sas_variable"] == feature), None)
        if entry:
            categories = list(set([cat['value'] for cat in entry['categories']]))
            categories.sort()
            if categories[0] != str(1):
                features_classification[feature] = 'numerical'

                print(f"Feature: {feature}")
                print(categories)

            else:
                features_classification[feature] = 'categorical'
        else:
            features_classification[feature] = 'unknown'

    # Handle missing values for numerical features
    continuous_columns = ['PHYSHLTH','MENTHLTH','POORHLTH',
                      'SLEPTIM1','WEIGHT2','HEIGHT3',
                      'MARIJAN1','ALCDAY4','AVEDRNK3','DRNK3GE5',
                      'MAXDRNKS','COPDSMOK','_PACKDAY',
                      ]

    X["_PACKDAY"] = X["_PACKDAY"].fillna(X["_PACKDAY"].median())

    X["COPDSMOK"] = X["COPDSMOK"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["COPDSMOK"].median())

    X["MAXDRNKS"] = X["MAXDRNKS"].replace({77:np.nan, 99:np.nan}).fillna(X["MAXDRNKS"].median())

    X["DRNK3GE5"] = X["DRNK3GE5"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["DRNK3GE5"].median())

    X["AVEDRNK3"] = X["AVEDRNK3"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["AVEDRNK3"].median())

    X["MARIJAN1"] = X["MARIJAN1"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["MARIJAN1"].median())

    mask_alc_week = X["ALCDAY4"].astype(float).between(101, 199)
    X.loc[mask_alc_week, "ALCDAY4"] = (X.loc[mask_alc_week, "ALCDAY4"] % 100) * 4
    mask_alc_month = X["ALCDAY4"].between(201, 299)  
    X.loc[mask_alc_month, "ALCDAY4"] = (X.loc[mask_alc_month, "ALCDAY4"] % 200)

    X["ALCDAY4"] = X["ALCDAY4"].replace({888:0, 777:np.nan, 999:np.nan}).fillna(X["ALCDAY4"].median())

    X["SLEPTIM1"] = X["SLEPTIM1"].replace({77:np.nan, 99:np.nan}).fillna(X["SLEPTIM1"].median())

    X["POORHLTH"] = X["POORHLTH"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["POORHLTH"].median())

    X["MENTHLTH"] = X["MENTHLTH"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["MENTHLTH"].median())

    X["PHYSHLTH"] = X["PHYSHLTH"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["PHYSHLTH"].median())
    
    X["WEIGHT2"] = X["WEIGHT2"].astype(float)
    mask_lb = X["WEIGHT2"].between(50, 776)
    X.loc[mask_lb, "WEIGHT2"] = X.loc[mask_lb, "WEIGHT2"] * 0.45359237

    mask_kg = X["WEIGHT2"].between(9023, 9352) 
    X.loc[mask_kg, "WEIGHT2"] = X.loc[mask_kg, "WEIGHT2"] - 9000

    mask_ft_in = X["HEIGHT3"].astype(float).between(200, 711)
    X.loc[mask_ft_in, "HEIGHT3"] = X.loc[mask_ft_in, "HEIGHT3"].apply(ft_in_to_cm)

    mask_kg = X["HEIGHT3"].between(9061, 9998)  
    X.loc[mask_kg, "HEIGHT3"] = X.loc[mask_kg, "HEIGHT3"] - 9000

    X["WEIGHT2"] = X["WEIGHT2"].replace({7777:np.nan, 9999:np.nan}).fillna(X["WEIGHT2"].median())

    X["HEIGHT3"] = X["HEIGHT3"].replace({7777:np.nan, 9999:np.nan}).fillna(X["HEIGHT3"].median())

    categorical_features = [feat for feat, info in features_classification.items() if info == 'categorical']

    X[categorical_features] = X[categorical_features].fillna(-1)
    X[categorical_features] = (X[categorical_features].astype('category').apply(lambda x: x.cat.codes))

    X = X.set_index('ID')

    scaler = StandardScaler()

    X[continuous_columns] = scaler.fit_transform(X[continuous_columns])

    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(12,6))
    mi_series.plot(kind='bar')
    plt.ylabel('Mutual Information')
    plt.title('Importance des features par rapport au label')
    plt.show()

    mi_cumsum = mi_series.cumsum()
    mi_total = mi_series.sum()

    threshold = 0.99 * mi_total
    features_99 = mi_cumsum[mi_cumsum <= threshold].index.tolist()

    print("Nombre de features conservées :", len(features_99))
    print("Features conservées :", features_99)

    X = X[features_99]

 
    y = df['TARGET']

    return X, y, features_99, scaler

def data_processing_test(df, feature_info, scaler) :

    parser = HTMLParser()
    html_snippet = "data/USCODE22_LLCP_102523.HTML"

    codebook = parser.parse_html_codebook(html_snippet)

    config = "features.yaml"

    with open(config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    

    X = df.drop(columns='TARGET')    
    y = df['TARGET'].astype(int)


    X = X[config['features']].copy()

    features_classification = {}

    for feature in config['features']:
        features_classification[feature] = {}
        entry = next((item for item in codebook if item["sas_variable"] == feature), None)
        if entry:
            categories = list(set([cat['value'] for cat in entry['categories']]))
            categories.sort()
            if categories[0] != str(1):
                features_classification[feature] = 'numerical'

                print(f"Feature: {feature}")
                print(categories)

            else:
                features_classification[feature] = 'categorical'
        else:
            features_classification[feature] = 'unknown'

    # Handle missing values for numerical features
    continuous_columns = ['PHYSHLTH','MENTHLTH','POORHLTH',
                      'SLEPTIM1','WEIGHT2','HEIGHT3',
                      'MARIJAN1','ALCDAY4','AVEDRNK3','DRNK3GE5',
                      'MAXDRNKS','COPDSMOK','_PACKDAY',
                      ]

    X["_PACKDAY"] = X["_PACKDAY"].fillna(X["_PACKDAY"].median())

    X["COPDSMOK"] = X["COPDSMOK"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["COPDSMOK"].median())

    X["MAXDRNKS"] = X["MAXDRNKS"].replace({77:np.nan, 99:np.nan}).fillna(X["MAXDRNKS"].median())

    X["DRNK3GE5"] = X["DRNK3GE5"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["DRNK3GE5"].median())

    X["AVEDRNK3"] = X["AVEDRNK3"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["AVEDRNK3"].median())

    X["MARIJAN1"] = X["MARIJAN1"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["MARIJAN1"].median())

    mask_alc_week = X["ALCDAY4"].astype(float).between(101, 199)
    X.loc[mask_alc_week, "ALCDAY4"] = (X.loc[mask_alc_week, "ALCDAY4"] % 100) * 4
    mask_alc_month = X["ALCDAY4"].between(201, 299)  
    X.loc[mask_alc_month, "ALCDAY4"] = (X.loc[mask_alc_month, "ALCDAY4"] % 200)

    X["ALCDAY4"] = X["ALCDAY4"].replace({888:0, 777:np.nan, 999:np.nan}).fillna(X["ALCDAY4"].median())

    X["SLEPTIM1"] = X["SLEPTIM1"].replace({77:np.nan, 99:np.nan}).fillna(X["SLEPTIM1"].median())

    X["POORHLTH"] = X["POORHLTH"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["POORHLTH"].median())

    X["MENTHLTH"] = X["MENTHLTH"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["MENTHLTH"].median())

    X["PHYSHLTH"] = X["PHYSHLTH"].replace({88:0, 77:np.nan, 99:np.nan}).fillna(X["PHYSHLTH"].median())
    
    X["WEIGHT2"] = X["WEIGHT2"].astype(float)
    mask_lb = X["WEIGHT2"].between(50, 776)
    X.loc[mask_lb, "WEIGHT2"] = X.loc[mask_lb, "WEIGHT2"] * 0.45359237

    mask_kg = X["WEIGHT2"].between(9023, 9352) 
    X.loc[mask_kg, "WEIGHT2"] = X.loc[mask_kg, "WEIGHT2"] - 9000

    mask_ft_in = X["HEIGHT3"].astype(float).between(200, 711)
    X.loc[mask_ft_in, "HEIGHT3"] = X.loc[mask_ft_in, "HEIGHT3"].apply(ft_in_to_cm)

    mask_kg = X["HEIGHT3"].between(9061, 9998)  
    X.loc[mask_kg, "HEIGHT3"] = X.loc[mask_kg, "HEIGHT3"] - 9000

    X["WEIGHT2"] = X["WEIGHT2"].replace({7777:np.nan, 9999:np.nan}).fillna(X["WEIGHT2"].median())

    X["HEIGHT3"] = X["HEIGHT3"].replace({7777:np.nan, 9999:np.nan}).fillna(X["HEIGHT3"].median())

    categorical_features = [feat for feat, info in features_classification.items() if info == 'categorical']

    X[categorical_features] = X[categorical_features].fillna(-1)
    X[categorical_features] = (X[categorical_features].astype('category').apply(lambda x: x.cat.codes))

    X = X.set_index('ID')

    X[continuous_columns] = scaler.fit_transform(X[continuous_columns])


    X = X[feature_info]

    return X
    




    
