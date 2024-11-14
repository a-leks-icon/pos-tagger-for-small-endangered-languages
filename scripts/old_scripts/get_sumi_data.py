#Script written by Aleksandr Schamberger as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt-University Berlin in the winter semester 2023/24.
#Created: 2024-01-28
#Latest Version: 2024-01-28

from corflow import fromElan
import glob
import re

def find_tiers(transcription,string):
    '''Returns a list with all tiers in the *transcription* whose names match the given regex *string*.'''
    found_tiers = []
    for tier in transcription:
        if re.search(string,tier.name):
            found_tiers.append(tier)
    return found_tiers


#ACTUAL OPERATIONS BEGIN HERE#
input_files = "../input_files/sumi_files/"
tagged_words_file_path = "../data/tagged_words_sumi.txt"

num_sentences = 0
eaf_files = glob.glob(input_files+"/*.eaf")
tagged_words = []
#ELAN files:
for file in eaf_files:
    print(f"File name: {file.replace(input_files,'')}")
    trans = fromElan.fromElan(file,encoding="utf-8")
    #Getting the transcribe, and respective word and pos tiers.
    for tr_tier in find_tiers(trans,"Transcribe-txt-nsm"):
        print(tr_tier.name)
        for ch_tier in tr_tier.children():
            if "Words-txt-nsm" in ch_tier.name:
                wd_tier = ch_tier
                break
        for ch_ch_tier in wd_tier.children()[0]:
            if "word-pos-en-cp" in ch_ch_tier.name:
                pos_tier = ch_ch_tier
                break
        #Getting pairs of words segments and pos tags.
        for tr_seg in tr_tier:
            tr_concat = tr_seg.content.replace(" ","").replace("-","")
            if wd_tier in tr_seg.childDict().keys():
                wd_concat = ""
                for wd_seg in tr_seg.childDict()[wd_tier]:
                    if pos_tier in wd_seg.childDict().keys():
                        pos_seg = wd_seg.childDict()[pos_tier][0]
                        tagged_words.append((wd_seg.content,pos_seg.content))
                    wd_concat += wd_seg.content
                    #Additionally, adding punctuation, if the comparison between concatenated word segments and transcribe segments is positive.
                    if tr_concat.startswith((wd_concat+",")):
                        wd_concat += ","
                        tagged_words.append((",","punct"))
                    elif tr_concat.startswith((wd_concat+".")):
                        wd_concat += "."
                        tagged_words.append((".","punct"))
        #Adding a dot, if the last transcribe segment does not end on a dot.
        if len(tr_tier) > 0:
            if not tr_tier.elem[-1].content.endswith("."):
                tagged_words.append((".","punct"))

#Saving all pairs of word-(pos)tag with ';' as a delimiter in a separate csv-file.
tagged_words = [(word,tag) for word,tag in tagged_words]
with open(tagged_words_file_path,"w") as file:
    for w,t in tagged_words:
        file.write(w+";"+t+"\n")