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

