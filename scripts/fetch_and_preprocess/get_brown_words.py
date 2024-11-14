#Script written by Aleksandr Schamberger (GitHub: https://github.com/a-leks-icon) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt Universität zu Berlin in the winter semester 2023/24.

#Created: 2024-05-09
#Latest Version: 2024-09-29

#Make sure to use 'nltk.download("brown")' and 'nltk.download("universal_tagset")'
#to download the necessary packages.
from nltk.corpus import brown
import numpy as np

def reduce_dim(data:list):
    '''Reduces the dimension of an Iterable **data** and returns it as a new list.
    
    Args:
        data: Iterable, whose elements are themselves Iterables.

    Returns:
        List.
    '''
    new_data = []
    for item in data:
        new_data += item
    return new_data

def create_dataset(sents:list[list[tuple[str,str]]],tok_limit:int|bool=False,seed:int|bool|float=True,end_tag:str="<E>") -> list:
    '''Creates a list of word-tag-pairs.

    First, pseudo randomly shuffles a list with lists representing sentences **sents**, whose elements are word-tag-pairs, with a pseudorandom **seed**. Second, adds every word-tag-pair to a new list as well as a tuple with two elements, whose values equal **end_tag**, after every sentence. If **tok_limit** is an integer, it defines the limit of added word-tag-pairs after which iterating through sentences stops and the new list is returned.

    Args:
        sents: List with lists representing sentences. Every sentence contains a number of tuples with two elements representing word-tag-pairs.
        tok_limit: Number limiting the number of word-tag-paris and therefore sentences added to the new list. If a boolean expression, no limit applies.
        seed: Number to initialize the pseudorandom number generator to randomly shuffle the list of sentences. Using the same number results in the same outcome for the final list of word-tag-pairs. By default, **seed** is True indicating that no random seed is given. If seed is False, no random seed is given and no random shuffling of **sents** takes place.
        end_tag: String representing the end tag used to encode the end of a sentence in a list of word-tag-pairs.

    Returns:
        List with word-tag-pairs.
    '''
    if isinstance(seed,bool):
        rng = np.random.default_rng()
        if seed == True:
            rng.shuffle(sents)
    elif isinstance(seed,(int,float)):
        rng = np.random.default_rng(seed)
        rng.shuffle(sents)

    tagged_sents = []
    if isinstance(tok_limit,bool):
        for sent in sents:
            for wd_tag in sent:
                tagged_sents.append(wd_tag)
            tagged_sents.append((end_tag, end_tag))
        return tagged_sents
    limit = 0
    for sent in sents:
        for wd_tag in sent:
            tagged_sents.append(wd_tag)
        tagged_sents.append((end_tag, end_tag))
        limit += len(sent)
        if limit >= tok_limit:
            return tagged_sents

#Path to the preprocessed data.
source_path = "../../data/preprocessed/"
#Name of the file with preprocessed data.
file_name = "brown_medium.txt"
#Name of the final file.
output_file = source_path+file_name

#Load the A16 subcorpus from the Brown Corpus.
#Get sentences as lists containing word-tag-pairs
#as tuples based on the Universal Dependency tagset.
brown_tagged_sents = list(brown.tagged_sents(categories="news",tagset="universal"))
#Sets a limit to how many word-tag-pairs are collected in the next step.
#Limit for Urum words: 18976 (without <E>).
#Limit for Urum morphs: 34817 (without <E>).
token_limit = 34817
#Pseudo randomly shuffles the sentences and creates a new list of
#word-tag-pairs but with added end tags after the end of a sentence.
brown_wd_tag_pairs = create_dataset(brown_tagged_sents,token_limit,False)

#Saving all pairs of word-tag with ';' as a delimiter in a separate csv-file.
with open(output_file,"w") as file:
    for w,t in brown_wd_tag_pairs:
        file.write(w+"\t"+t+"\n")

print(f"Done.")