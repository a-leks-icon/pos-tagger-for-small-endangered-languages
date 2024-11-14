#Script written by Aleksandr Schamberger (GitHub: https://github.com/a-leks-icon) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt Universität zu Berlin in the winter semester 2023/24.

#Created: 2024-01-28
#Latest Version: 2024-09-29

from corflow.fromElan import fromElan
import glob
import re

#Path of the Urum .eaf-files.
source_path = "../../data/source/doreco_urum1249_extended/"
#Name of the file with preprocessed data.
output_file = "../../data/preprocessed/urum_morphs.txt"
#Get all .eaf-files in a list.
eaf_files = glob.glob(source_path+"/*.eaf")
#Create an empty list to collect morph-tag-pairs.
tagged_morphs = []
#Strings, based on which to ignore segments.
bad_strings = ["", "****", "<p:>"]
#Accepted DoReCo lables.
doreco_labels = ["<<fm", "<<pr", "<<ui"]
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
                    #Initialize a temporary list to to collect morph-tag-pairs and to
                    #check whether morphs constitute a well-formed word.
                    tagged_morphs_temp = []
                    #Iterate through every morph break segment and its pos segment.
                    for mb_seg in wd_seg.childDict()[mb_tier]:
                        if pos_tier in mb_seg.childDict():
                            pos_seg = mb_seg.childDict()[pos_tier][0]
                            #Skip, if it's empty, the DoReCo label '****' or a silent pause.
                            if pos_seg.content in bad_strings:
                                continue
                            #Remove any whitespace.
                            pos = pos_seg.content.replace(" ","")
                            mb = mb_seg.content.replace(" ","")
                            #Clean pos tag, if it has a typo or incoherence.
                            if pos in clean_pos_dict.keys():
                                pos = clean_pos_dict[pos]
                            if (mb.startswith("-")) & (pos == "X"):
                                pos = "-"+pos
                            #Collect the morph-tag-pair.
                            tagged_morphs_temp.append((mb,pos))
                    #Do not add morph-tag-pairs, if the root they belong to
                    #is not well-formed (its pos tag equals "X").
                    well_formed = True
                    for mb,pos in tagged_morphs_temp:
                        if (pos in ["xxx", "X", "?"]) & (not mb.startswith("-")):
                            well_formed = False
                    #Otherwise, add the morph-tag-pairs.
                    if well_formed:
                        for mb_pos in tagged_morphs_temp:
                            tagged_morphs.append(mb_pos)

        #After having iterated through every morph and its pos tag given a word and ref seg,
        #add the end tag to the list of morph-tag-pairs to encode the end of a sentence.
        if tagged_morphs[-1][-1] != "<E>":
            tagged_morphs.append(("<E>","<E>"))

#Some morph-tag-pairs were wrong and could not be changed
#by just adding a key-value-pair to the dictionary.
for ind,(mb,tag) in enumerate(tagged_morphs):
    if (mb == "-lari") & (tag == "-PN"):
        tagged_morphs[ind] = (mb,"-prs")
    elif mb == "-to":
        tagged_morphs[ind] = ("to","PRT")
    elif mb == "-što":
        tagged_morphs[ind] = ("što","PN")
    elif (mb == "o") & (tag == "-PN"):
        tagged_morphs[ind] = (mb,"PN")
    elif mb == "-kim":
        tagged_morphs[ind] = ("kim","PN")

#Save every collected morph-tag-pair with ';' as a delimiter in a .csv-file.
with open(output_file,"w") as file:
    for w,t in tagged_morphs:
        file.write(w+";"+t+"\n")

print(f"Done.")
