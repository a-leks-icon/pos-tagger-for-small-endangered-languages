#Script written by Aleksandr Schamberger (GitHub: https://github.com/a-leks-icon) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt Universität zu Berlin in the winter semester 2023/24.

#Created: 2024-05-09
#Latest Version: 2024-09-29

import pandas as pd

def get_csv_as_list(file_path:str,separator:str) -> list[tuple[str,str]]:
    '''Imports a .csv-file as a list.

    Imports a .csv-file **file_path** with **separator** as the delimiter and returns it as a list with tuples. Every tuple represents a row/line and every element in a tuple represents a unique value in a column.
    
    Args:
        file_path: Path to the .csv-file.
        separator: Delimiter used to separate datapoints in the .csv-file.

    Returns:
        A list of tuples.
    '''
    tagged_words = pd.read_csv(file_path,sep=separator,header=None,on_bad_lines="warn")
    tagged_words = tagged_words.dropna()
    tagged_words_l = tagged_words.values.tolist()
    return [(i[0],i[1]) for i in tagged_words_l]

def get_sent_num(wd_tag_pairs:list[tuple[str,str]]) -> int:
    '''Return the number of sentences.

    Compute the number of sentences for a list of word-tag-pairs **wd_tag_pairs** based on the end tag `<E>` and whether the end tag's previous tag is not an end tag.

    Args:
        wd_tag_pairs: List of word-tag-pairs (tuples).

    Returns:
        Integer representing the number of sentences.
    '''
    sents_num = 0
    tokens = []
    for wd,tag in wd_tag_pairs:
        if tag != "<E>":
            tokens.append(tag)
        elif tokens:
            sents_num += 1
            tokens = []
    return sents_num

#START OPERATIONS#

#Path to preprocessed data.
data_path = "../../data/preprocessed/"
#File to be processed.
file = "brown_medium.txt"
#Get word-tag-pairs from the respective file as tuples inside a list.
wd_tag_pairs = get_csv_as_list(data_path+file,"\t")
#List of unique pos tags.
tags = list(set([tag for wd,tag in wd_tag_pairs]))
#Set of word types.
words = set([wd for wd,tag in wd_tag_pairs])
#Create a dictionary with pos tags as keys.
tags_dict = dict.fromkeys(tags)
#Iterate through every key (pos tag) in the dictionary
#and set the key's value equal to a tag's token frequency
for tag in tags_dict:
    num = len([t for wd,t in wd_tag_pairs if t == tag])
    tags_dict[tag] = num
#Sort the dictionary keys based on their values in descending order.
tags_dict = {k:v for k,v in sorted(tags_dict.items(),key=lambda item: item[1], reverse=True)}
#Get the number of sentences.
sents_num = get_sent_num(wd_tag_pairs)
print(tags_dict)
print(f"Total number of tokens: {len(wd_tag_pairs)-sents_num}")
print(f"Number of unique pos tags: {len(tags_dict)}")
print(f"Number of word/morph types: {len(words)}")
print(f"Number of sentences: {sents_num}")
print(f"Average number of tokens per sentence: {round((len(wd_tag_pairs)-sents_num)/sents_num,2)}")
