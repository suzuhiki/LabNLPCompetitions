import pandas as pd
import numpy as np
import sys

class DefaultLoader:
    def loaddf(self, text_path:str, label_path:str) -> object:
        text_list = self.loadtext(text_path)
        label_list = np.loadtxt(label_path, dtype=int)
        
        if len(text_list) != len(label_list):
            print(f"2つのファイルから読み込んだリストの長さが一致しません：{text_path},{label_path}")
            sys.exit(1)
        
        df = pd.DataFrame({"text":text_list, "labels":label_list})
        return df
        
    def loadtestdf(self, text_path:str) -> object:
        text_list = self.loadtext(text_path)
        df = pd.DataFrame({"text":text_list})
        return df
        
    def loadtext(self, text_path:str) -> list:
        with open(text_path, "r") as f:
            data = f.read().splitlines()
        return data
            