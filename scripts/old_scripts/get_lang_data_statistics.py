#Script written by Aleksandr Schamberger as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt-University Berlin in the winter semester 2023/24.
#Created: 2024-05-26
#Latest Version: 2024-05-26

import pandas as pd
import glob

def get_csv_as_list(file_path,separator):
    '''Loads a csv file *file_path* with *separator* as the delimiter and returns it as a list with tuples, in which every tuple represents a row and every value in a tuple represents a value of a column.'''
    tagged_words = pd.read_csv(file_path,sep=separator,header=None)
    tagged_words = tagged_words.dropna()
    tagged_words_l = tagged_words.values.tolist()
    return [(i[0],i[1]) for i in tagged_words_l]

def get_sent_from_wd_tag_pairs(wd_tag_pairs,wd_end,tag_end):
    '''Returns a list containing sentences, where a sentence is represented as a list containing any number of word-tag-pairs (as tuples) from a given list of word-tag-pairs *wd_tag_pairs* (as tuples) except for those pairs, whose word equals *wd_end* and whose tag equals *tag_end*. These pairs are only used to encode the end of a sentence and the beginning of a following sentence in *wd_tag_pairs*.'''
    tagged_sentences = []
    single_sentence = []
    for wd,tag in wd_tag_pairs:
        if (wd != wd_end) & (tag != tag_end):
            single_sentence.append((wd,tag))
        elif single_sentence:
            tagged_sentences.append(single_sentence)
            single_sentence = []
    return tagged_sentences

def get_tags(data):
    tags = list(set([tag for wd,tag in data]))
    return tags

###
data_path = "../data/"
lang_files = glob.glob(data_path+"**",recursive=True)
for file in lang_files[1:]:
    file_name = file.replace(data_path,"")
    print(f"file: {file_name}")
    if file_name.startswith("brown"):
        sep = "\t"
    else:
        sep = ";"
    data = get_csv_as_list(file,sep)
    data_sents = get_sent_from_wd_tag_pairs(data,"<E>","<E>")
    print(f"sentences: {len(data_sents)}")
    print(f"Tokens: {len(data)}")
    print(get_tags(data))
    print(f"POS tags amount: {len(get_tags(data))}")
    print("---")