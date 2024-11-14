#Script written by Aleksandr Schamberger (GitHub: https://github.com/a-leks-icon) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt Universität zu Berlin in the winter semester 2023/24.

#Created: 2024-01-28
#Latest Version: 2024-09-29

from corflow.fromElan import fromElan
import glob
import re

#Path of the Urum .eaf-files.
source_path = "../../data/source/doreco_urum1249_extended/"
#Name of the file with preprocessed data.
output_file = "../../data/preprocessed/urum_words.txt"
#Get all .eaf-files in a list.
eaf_files = glob.glob(source_path+"/*.eaf")
#Create an empty list to collect word-tag-pairs.
tagged_words = []
#Strings, based on which to ignore segments.
bad_strings = ["", "****", "<p:>"]
#Dictionary to change those pos tags containing typos or incoherences.
clean_pos_dict = {
    "PN-case": "PN",
    "Pn.": "Pn.A",
    "PN?": "PN",
    "VV": "V",
    "REF.PN": "PN",
    "Q\t": "Q",
    "N-num": "N",
    "INT": "INTJ",
    "NV": "V",
    "AN": "N",
    "INF": "V",
    #"X": "-X",
    "-tma": "-tam",
    "-tam-V": "-tam",
    "-tamA": "A",
    "-tam-tam": "-tam",
    "-tam-prs": "-prs",
    "tam-V": "tam",
    "-prsQ": "Q",
    "-cases": "-case",
    "case": "-case",
    "-caseV": "V",
    "allow": "V",
    "-N?": "-N",
    "-PST": "-V",
    "-pts": "-prs",
    "-nam": "-num",
    "v": "V",
    "-inf": "-fin",
    "-PRT": "PRT"
}

#Iterate through every .eaf-file.
for file in eaf_files:
    #print(f"File name: {file.replace(source_path,'')}")
    #Create a transcription object.
    trans = fromElan(file,encoding="utf-8")
    #Get the reference tier and its respective word, morph break and and pos tier:
    for ref_tier in trans.findAllName("ref@"):
        for ch_tier in ref_tier.children():
            if ch_tier.name.startswith("wd@"):
                wd_tier = ch_tier
                break
        mb_tier = wd_tier.children()[0]
        for ch_tier in mb_tier.children():
            if ch_tier.name.startswith("ps@"):
                pos_tier = ch_tier
                break
    #Accepted DoReCo lables.
    doreco_labels = ["<<fm", "<<pr", "<<ui"]
    #Iterate through every reference segment on the ref tier.
    for ref_seg in ref_tier:
        #Skip, if it's a silent pause.
        if ref_seg.content == "<p:>":
            continue
        if wd_tier in ref_seg.childDict():
            #Iterate through every word segment on the wd tier.
            for wd_seg in ref_seg.childDict()[wd_tier]:
                word = wd_seg.content
                for label in doreco_labels:
                    if word.startswith(label):
                        word = word.removeprefix(label).replace(">","")
                #Skip, if the word starts with an inappropriate DoReCo label, if
                #it's empty, if it's the DoReCo label '****' or a silent pause.
                if (word.startswith("<<")) | (word in bad_strings):
                        continue
                if mb_tier in wd_seg.childDict():
                    #Iterate through every morph break segment and its pos segment.
                    for mb_seg in wd_seg.childDict()[mb_tier]:
                        mb = mb_seg.content
                        if pos_tier in mb_seg.childDict():
                            pos_seg = mb_seg.childDict()[pos_tier][0]
                            #Skip, if it's empty, the DoReCo label '****' or a silent pause.
                            if pos_seg.content in bad_strings:
                                continue
                            #Remove any whitespace.
                            pos = pos_seg.content.replace(" ","")
                            #Clean pos tag, if it has a typo or incoherence.
                            if pos in clean_pos_dict.keys():
                                pos = clean_pos_dict[pos]
                            #Some morph-tag-pairs were wrong and could not be changed
                            #by just adding a key-value-pair to the dictionary.
                            elif mb == "-to":
                                pos = "PRT"
                            elif mb == "-što":
                                pos = "PN"
                            elif (mb == "o") & (pos == "-PN"):
                                pos = "PN"
                            elif mb == "-kim":
                                pos = "PN"
                            #Continue only, if the first character of the pos tag
                            #starts with an uppercase letter (except for 'X').
                            if re.search("[A-WY-Z]",pos[0]):
                                #Remove leading and trailing whitespaces.
                                word = word.strip()
                                #Collect the word-tag-pair.
                                tagged_words.append((word,pos))
                                break
        #After having iterated through every word and its pos tag given a ref seg,
        #add the end tag to the list of word-tag-pairs to encode the end of a sentence.
        if tagged_words[-1][-1] != "<E>":
            tagged_words.append(("<E>","<E>"))

#Save every collected word-tag-pair with ';' as a delimiter in a .csv-file.
with open(output_file,"w") as file:
    for w,t in tagged_words:
        file.write(w+";"+t+"\n")

print(f"Done.")
