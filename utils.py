import re, json, html
import pandas as pd
import numpy as np

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
    s = str(s).zfill(3)  
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
# data_reduced[categorical_features] = data_reduced[categorical_features].fillna(-1)

# data_reduced[categorical_features] = data_reduced[categorical_features].astype('category')


