#Script written by Aleksandr Schamberger (GitHub: JLEKS) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt-University Berlin in the winter semester 2023/24.
#Created: 2024-05-08
#Latest Version: 2024-05-09

'''
This script goes through every txt file found in data and checks whether elements (words or morphs) appearing with multiple tags do so in an unbalanced way (meaning that they appear with the most common tag way more times than they do with the second most common tag). If so, their tags get normalized to the one they occur most often.
'''

import pandas as pd
import glob

def get_csv_as_list(file_path,separator):
    '''Loads a csv file *file_path* with *separator* as the delimiter and returns it as a list with tuples, in which every tuple represents a row and every value in a tuple represents a value of a column.'''
    tagged_words = pd.read_csv(file_path,sep=separator,header=None)
    tagged_words = tagged_words.dropna()
    tagged_words_l = tagged_words.values.tolist()
    return [(i[0],i[1]) for i in tagged_words_l]

data = "../data/"
files = glob.glob(data+"/*.txt")

for enum,file in enumerate(files):
    print(f"Processing file: {file.replace(data,'')} | enum: {enum}")
    el_tag_l = get_csv_as_list(file,";")
    els = list(set([el for el,tag in el_tag_l]))
    el_tag_dict = {el: [] for el in els}
    print(f"Unique items: {len(el_tag_dict)}")
    for el,tag in el_tag_l:
        el_tag_dict[el].append(tag)
    for key,val in el_tag_dict.items():
        new_val = []
        tags = list(set(val))
        for tag in tags:
            new_val.append((len([t for t in val if t == tag]),tag))
        el_tag_dict[key] = new_val
    #el_tag_dict = {key: val for key,val in sorted(el_tag_dict.items(), key=lambda item: item[1], reverse=True)}
    change_pair = []
    for key,val in el_tag_dict.items():
        if len(val) > 1:
            _val = sorted(val,reverse=True)
            if _val[0][0] > _val[1][0]+500:
                change_pair.append((key,_val[0][1]))
                print(f"el: {key} | tags: {val}")
            elif len(val) > 4:
                print(f"x: el: {key} | tags: {val}")
    if change_pair:
        continue
        df = pd.read_csv(file,sep=";",header=None)
        for el,tag in change_pair:
            for label,row in df.iterrows():
                if row[0] == el:
                    row[1] = tag
        new_file = file.removesuffix(".txt") + "_modified.txt"
        df.to_csv(new_file,sep=";",index=False,header=False)

