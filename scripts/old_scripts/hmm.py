#Script written by Aleksandr Schamberger as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt-University Berlin in the winter semester 2023/24.
#Script is partly based on the blog post from MyGreatLearning <https://www.mygreatlearning.com/blog/pos-tagging/#sh2> (link lastly opened on 2024-01-28).
#Created: 2024-01-06
#Latest Version: 2024-02-05

import re
import pandas as pd
import numpy as np
import math
from operator import itemgetter
import tracemalloc


def find_tiers(transcription,string):
    '''Returns a list with all tiers in the *transcription* whose names match the given regex *string*.'''
    found_tiers = []
    for tier in transcription:
        if re.search(string,tier.name):
            found_tiers.append(tier)
    return found_tiers


def split_train_test(data,size=0.8,seed=False):
    '''Returns a tuple with two lists, the first being the training data and the second being the test data based on randomly shuffling the *data* with a given random *seed* (by default no seed is given). The size of the training data equals *size* times its length (by default 0.8*length of *data*). The rest of *data* amounts to the test data.'''
    if isinstance(seed,bool):
        rng = np.random.default_rng()
    elif isinstance(seed,int):
        rng = np.random.default_rng(seed)
    split_num = math.ceil(len(data)*size)
    random_data = data
    rng.shuffle(random_data)
    train = random_data[:split_num]
    test = random_data[split_num:]
    return (train,test)

def reduce_dim(data:list):
    '''Reduces the dimension of a list *data* and returns it as a new list.'''
    new_data = []
    for item in data:
        new_data += item
    return new_data

def get_csv_as_list(file_path,seperator,):
    '''Loads a csv file from *file_path* with *sep* as the delimiter and returns it as list with tuples, in which every tuple represents a row and every value in a tuple represents a value of a column.'''
    tagged_words = pd.read_csv(file_path,sep=seperator,header=None)
    tagged_words = tagged_words.dropna()
    tagged_words_l = tagged_words.values.tolist()
    return [(i[0],i[1]) for i in tagged_words_l]

def get_sent_for_wd_tag_pairs(wd_tag_pairs:list):
    '''Returns a list of sentences, where a sentence is represented as a list containing any number of word-tag-pairs (as tuples) from a given list of word-tag-pairs *wd_tag_pairs* (as tuples). If *wd_tag_pairs* contains list of word-tag-pairs rather than tuples, the former get transformed to tuples.'''
    pairs = wd_tag_pairs
    if any(isinstance(pair,list) for pair in wd_tag_pairs):
        pairs = [(i[0],i[1]) for i in wd_tag_pairs]
    tagged_sentences = []
    single_sentence = []
    for wd,tag in pairs:
        single_sentence.append((wd,tag))
        if (wd == ".") & (tag == "punct"):
            tagged_sentences.append(single_sentence)
            single_sentence = []
    return tagged_sentences

def get_sent_for_wd_tag_pairs_urum(wd_tag_pairs:list):
    '''Returns a list of sentences, where a sentence is represented as a list containing any number of word-tag-pairs (as tuples) from a given list of word-tag-pairs *wd_tag_pairs* (as tuples). If *wd_tag_pairs* contains list of word-tag-pairs rather than tuples, the former get transformed to tuples.'''
    pairs = wd_tag_pairs
    if any(isinstance(pair,list) for pair in wd_tag_pairs):
        pairs = [(i[0],i[1]) for i in wd_tag_pairs]
    tagged_sentences = []
    single_sentence = []
    for wd,tag in pairs:
        if wd != "<E>":
            single_sentence.append((wd,tag))
        else:
            tagged_sentences.append(single_sentence)
            single_sentence = []

    return tagged_sentences

def get_emission_probability_base(word:str,tag:str,data):
    '''Returns for calculating the emission probability of a *word* given a *tag* from a list of word-tag-pairs (tuples) *data* a tuple with two ints.'''
    #P(word|tag) = (P(word) ∩ P(tag))/P(tag)
    #P(word|tag) = C(<tag,word>)/C(<tag,X>)
    pairs_with_tag = [(wd,t) for wd,t in data if t == tag]
    tag_count = len(pairs_with_tag)
    word_tag_count = len([wd for wd,t in pairs_with_tag if wd == word])
    return (word_tag_count,tag_count)

def get_emission_probability(training_sentences):
    '''Returns a dictionary with the emission probability of every word per tag given a list *training_sentences*, whose lists contain single sentences of the training data.'''
    train_data = reduce_dim(training_sentences)
    unique_tags = list(set([tag for wd,tag in train_data]))
    unique_words = list(set([wd for wd,tag in train_data]))
    emis_prob_dict = {}
    for wd in unique_words:
        emis_prob_dict[wd] = {}
        for tag in unique_tags:
            wd_tag_count,tag_count = get_emission_probability_base(wd,tag,train_data)
            if (wd_tag_count == 0) | (tag_count == 0):
                #emis_prob_dict[wd][tag] = 0
                continue
            else:
                emis_prob_dict[wd][tag] = wd_tag_count/tag_count
    return emis_prob_dict


def get_transition_probability_base(tag1,tag2,data):
    '''Returns for calculating the transition probability of a pair of tags <*tag1*,*tag2*> (*tag1* occurs right before *tag2*) from a list of word-tag-pairs (tuples) *data* a tuple with two ints.'''
    #P(tag2|tag1) = C(<tag1,tag2>)/C(<tag1,X>)
    tags = [tag for word,tag in data]
    count_tag1_tag2 = 0
    for ind in range(len(tags)-1):
        if (tags[ind] == tag1) & (tags[ind+1] == tag2):
            count_tag1_tag2 += 1
    #count_tag1_tag2 = len([tag for ind,tag in enumerate(tags) if (tag == tag1) & (tags[ind+1] == tag2)])
    count_tag1_x = len([tag for tag in tags if tag == tag1])
    return (count_tag1_tag2,count_tag1_x)

def get_transition_probability(training_sentences):
    '''Return a pandas DataFrame representing the transition probability of all unique (pos) tags (as well as the added start- and end-sentence tags '<S>' and '<E>') of a given list *training_sentences*, whose lists contain single sentences of the training data. The rows represent the first (left) tag (of a bigram), and the columns the second (right) tag.'''
    new_sentences = training_sentences
    for sent in new_sentences:
        sent.insert(0,("<S>","<S>"))
        sent.append(("<E>","<E>"))
    new_data = reduce_dim(new_sentences)
    unique_tags = list(set([tag for wd,tag in new_data]))
    trans_prob_arr = np.zeros((len(unique_tags),len(unique_tags)))
    for ind1,t1 in enumerate(unique_tags):
        for ind2,t2 in enumerate(unique_tags):
            if t1 == "<E>":
                trans_prob_arr[ind1,ind2] = 0
            elif t2 == "<S>":
                trans_prob_arr[ind1,ind2] = 0
            else:
                tag1_tag2,tag1_x = get_transition_probability_base(t1,t2,new_data)
                trans_prob_arr[ind1,ind2] = tag1_tag2/tag1_x
    return pd.DataFrame(trans_prob_arr,columns=unique_tags,index=unique_tags)

def hmm_algorithm(train_sentences,test_sentences):
    ''''''
    trans_prob_df = get_transition_probability(train_sentences)
    emis_prob_dict = get_emission_probability(train_sentences)
    test_sents = test_sentences
    for sent in test_sents:
        sent.insert(0,("<S>","<S>"))
        sent.append(("<E>","<E>"))
    correct_sent_predictions = 0
    #correct_tag_predictions = 0
    for sent in test_sents:
        #print(f"test sent: {sent}")
        sent_unpredictable = False
        paths = []
        for ind,wd_tag_pair in enumerate(sent[1:]):
            wd = wd_tag_pair[0]
            current_paths = []
            #If a word occurs in the test but not in the dict of the train sentences, the sentence cannot be predicted; therefore we stop the prediction. It failed.
            if not wd in emis_prob_dict.keys():
                sent_unpredictable = True
                break
            elif ind == (len(sent)-1):
                for tags_l,p in paths:
                    tag = tags_l[-1]
                    trans_p = trans_prob_df.at[tag,"<E>"]
                    tags_l.append("<E>")
                    p = p * trans_p
            elif (ind > 0) and (paths):
                for tag_key,em_p in emis_prob_dict[wd].items():
                    for ind2,(tags_l,p) in enumerate(paths):
                        trans_p = trans_prob_df.at[tags_l[-1],tag_key]
                        if trans_p == 0:
                            continue
                        current_tags = tags_l + [tag_key]
                        current_p = p * em_p * trans_p
                        current_paths.append((current_tags,current_p))
            elif ind == 0:
                for tag_key,em_p in emis_prob_dict[wd].items():
                    trans_p = trans_prob_df.at["<S>",tag_key]
                    if trans_p == 0:
                        continue
                    current_tags = ["<S>",tag_key]
                    current_p = trans_p * em_p
                    current_paths.append((current_tags,current_p))
            #If during the calculation of all paths p value, no path has a trans_p > 0, therefore, no path has any predictive value, the sentence is not predictable and therefore skipped.
            elif not paths:
                sent_unpredictable = True
                break

            paths = current_paths

        #Check, whether a word was in the test but not in the train sentences. If so, skip this sentence.
        if (sent_unpredictable) | (not paths):
            continue

        #print(f"computed paths: {paths}")
        optimal_path = max(paths,key=itemgetter(1))
        #print(f"optimal path: {optimal_path}")
        actual_path = [tag for wd,tag in sent]
        #Check, whether the prediction is correct or not.
        if optimal_path[0] == actual_path:
            correct_sent_predictions += 1
        #for ind in range(len(optimal_path[0])):
            #if optimal_path[0][ind] == actual_path[ind]:
                #correct_tag_predictions += 1


    return correct_sent_predictions/len(test_sentences)#),(correct_tag_predictions/len(reduce_dim(test_sentences))))


tracemalloc.start()
#ACTUAL OPERATIONS BEGIN HERE#
#tagged_words_file_path = "../data/tagged_words_urum_words.txt"
tagged_words_file_path = "../data/tagged_words_urum_words.txt"

#Get the word-tag-pairs from the respective file as tuples inside a list.
tagged_words_l = get_csv_as_list(tagged_words_file_path,";")

#Put word-tag-pairs into lists representing sentences and put them in one list.
tagged_sentences = get_sent_for_wd_tag_pairs_urum(tagged_words_l)
#tagged_sentences = get_sent_for_wd_tag_pairs(tagged_words_l)

#Create the training and test data randomly (optionally with a seed) by default in the ration 8:2 (train:test) from the list of tagged sentences.
train_sentences,test_sentences = split_train_test(tagged_sentences,seed=469283984701)
train_data = reduce_dim(train_sentences)

#Apply the hmm algorithm and print the accuracy of correctly predicted pos tags for the test sentences in percent.
hmm_accuracy = hmm_algorithm(train_sentences,test_sentences)
print(f"The hmm's accuracy is {hmm_accuracy*100}%.")
print(f"memory: {tracemalloc.get_traced_memory()}")
tracemalloc.stop()

'''
tr_sentences = [
    [("das","d"), ("beten","n"), ("hilft","v"), ("den","d"), ("mönchen","n"), (".","punct")],
    [("die","d"), ("mönche","n"), ("helfen", "v"), ("den","d"), ("pfarrern","n"), (".","punct")],
    [("die","d"), ("pfarrer","n"), ("beten","v"), (".","punct")],
    [("das","d"), ("treffen","n"), ("gefällt","v"), ("den","d"), ("mönchen","n"), (".","punct")],
    [("die","d"), ("pfarrer","n"), ("treffen","v"), ("die","d"), ("mönche","n"), (".","punct")],
    [("das","pro"), ("hilft","v"), ("den","d"), ("pfarrern","n"), (".","punct")],
    [("den","d"), ("mönchen","n"), ("gefällt","v"), ("das","pro"), (".","punct")],
    [("denen","pro"), ("helfen","v"), ("mönche", "n"), (".", "punct")]
]

te_sentences = [
    [("das","d"), ("beten","n"), ("hilft","v"), ("den","d"), ("pfarrern","n"), (".","punct")],
    [("das","pro"), ("gefällt","v"), ("den","d"), ("mönchen","n"), (".","punct")],
    [("die","d"), ("mönche","n"), ("beten","v"), (".","punct")],
    [("pfarrer", "n"), ("helfen", "v"), ("denen", "pro"), (".", "punct")]
]
'''
