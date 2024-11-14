#Script written by Aleksandr Schamberger (GitHub: JLEKS) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt-University Berlin in the winter semester 2023/24.
#Script is partly based on the blog post from MyGreatLearning <https://www.mygreatlearning.com/blog/pos-tagging/#sh2> (link lastly opened on 2024-01-28).
#Created: 2024-05-20
#Latest Version: 2024-05-20
#Version 4_rng1: Missing words get assigned one random tag, which occurs in the training data, with a probability of 1.

import pandas as pd
import numpy as np
import math
from operator import itemgetter

def get_csv_as_list(file_path,separator):
    '''Loads a csv file *file_path* with *separator* as the delimiter and returns it as a list with tuples, in which every tuple represents a row and every value in a tuple represents a value of a column.'''
    tagged_words = pd.read_csv(file_path,sep=separator,header=None,on_bad_lines="warn")
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

def split_train_test(data,size=0.8,seed=False):
    '''Returns a tuple with two lists, the first being the training data and the second being the test data based on randomly shuffling the *data* with a given random *seed* (by default no seed is given). The size of the training data equals *size* times its length (by default 0.8*length of *data*). The rest of *data* amounts to the test data. *data* itself is a list containing lists, each representing a sentence and containing pairs (tuples) of words and (pos)-tags.'''
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

def get_emission_probability(train_data):
    '''Returns a pandas data frame representing the emission probability of all unique word-tag-pairs of a given list of word-tag-pairs *train_data*. In the final data frame, the rows represent the words and the columns the (pos) tags.'''

    def get_emission_probability_base(word:str,tag:str,data):
        '''Returns for calculating the emission probability of a *word* given a *tag* from a list of word-tag-pairs (tuples) *data* a tuple with two ints.'''
        #P(word|tag) = (P(word) ∩ P(tag))/P(tag)
        #P(word|tag) = C(<tag,word>)/C(<tag,X>)
        pairs_with_tag = [(wd,t) for wd,t in data if t == tag]
        tag_count = len(pairs_with_tag)
        word_tag_count = len([wd for wd,t in pairs_with_tag if wd == word])
        return word_tag_count/tag_count

    unique_words = tuple(set([wd for wd,tag in train_data]))
    unique_tags = list(set([tag for wd,tag in train_data]))
    emis_prob_array  = np.zeros((len(unique_words),len(unique_tags)))
    for wd_ind,wd in enumerate(unique_words):
        relevant_unique_tags = tuple(set([tag for word,tag in train_data if word == wd]))
        for tag in relevant_unique_tags:
            tag_ind = unique_tags.index(tag)
            emis_prob_array[wd_ind,tag_ind] = get_emission_probability_base(wd,tag,train_data)
    return pd.DataFrame(emis_prob_array,index=unique_words,columns=unique_tags)

def get_tag_types(word:str,data):
    '''Returns a list with unique tags for a given *word* of a list of tuples of word-tag pairs *data*.'''
    unique_tags = set()
    for wd,tag in data:
        if wd == word:
            unique_tags.add(tag)
    return list(unique_tags)

def get_transition_probability(training_sentences):
    '''Return a pandas data frame representing the transition probability of all unique (pos) tags (as well as the added start- and end-sentence tags '<S>' and '<E>') of a given list *training_sentences*, whose lists contain single sentences of the training data. In the final data frame, the rows represent the first (left) tag (of a bigram), and the columns the second (right) tag.'''

    def get_transition_probability_base(tag1,tag2,data):
        '''Returns for calculating the transition probability of a pair of tags <*tag1*,*tag2*> (*tag1* occurs right before *tag2*) from a list of word-tag-pairs (tuples) *data* a tuple with two integers: 1. The number of times the bigram <tag1,tag2> occurs in *data*. 2. The number of times the unigram <tag1> occurs in *data*.'''
        #P(tag2|tag1) = C(<tag1,tag2>)/C(<tag1,X>)
        tags = [tag for word,tag in data]
        count_tag1_tag2 = 0
        for ind in range(len(tags)-1):
            if (tags[ind] == tag1) & (tags[ind+1] == tag2):
                count_tag1_tag2 += 1
        count_tag1_x = len([tag for tag in tags if tag == tag1])
        return (count_tag1_tag2,count_tag1_x)

    new_sentences = [sent.copy() for sent in training_sentences]
    for sent in new_sentences:
        sent.insert(0,("<S>","<S>"))
        sent.append(("<E>","<E>"))
    new_data = reduce_dim(new_sentences)
    unique_tags = list(set([tag for wd,tag in new_data]))
    trans_prob_array = np.zeros((len(unique_tags),len(unique_tags)))
    for ind1,t1 in enumerate(unique_tags):
        for ind2,t2 in enumerate(unique_tags):
            if t1 == "<E>":
                trans_prob_array[ind1,ind2] = 0
            elif t2 == "<S>":
                trans_prob_array[ind1,ind2] = 0
            else:
                tag1_tag2,tag1_x = get_transition_probability_base(t1,t2,new_data)
                trans_prob_array[ind1,ind2] = tag1_tag2/tag1_x
    return pd.DataFrame(trans_prob_array,columns=unique_tags,index=unique_tags)

def get_relevant_tags(data,freq=10):
    '''Returns a list containing only those (pos) tags from a given list of element-tag-pairs *data*, which occur at least *freq* (default: 10) times.'''
    tags = list(set([tag for el,tag in data]))
    tags_dict = dict.fromkeys(tags)
    rem_key = []
    for tag in tags_dict:
        num = len([t for el,t in data if t == tag])
        if num < freq:
            rem_key.append(tag)
    for key in rem_key:
        del tags_dict[key]
    return list(tags_dict.keys())

def viterbi_hmm_algorithm(train_sentences,test_sentences):
    '''Creates an hmm model of *train_sentences* and predicts *test_sentences* based on that model. Returns a tuple with its first value being the accuracy of the hmm model when predicting sentences and its second value being the accuracy of the hmm model when predicting single (pos) tags.'''
    print(f"Training.")
    trans_prob_df = get_transition_probability(train_sentences)
    test_sents = test_sentences
    train_data = reduce_dim(train_sentences)
    emis_prob_df = get_emission_probability(train_data)
    print(f"Predicting.")
    #Iterating over every tag, even if a word does not appear with that tag at all, seems to be slower than first getting only the relevant tags and only then iterating through those.
    all_tags = get_relevant_tags(train_data)
    sentences_correctly_predicted = 0
    words_correctly_predicted = 0
    skipped_sents = 0
    rng = np.random.default_rng()
    for c,sent in enumerate(test_sents):
        #print(f"{c+1}/{len(test_sents)}: {sent}")
        paths = []
        for ind,wd_tag_pair in enumerate(sent):

            #'''
            if len(paths) > 200_000:
                skipped_sents += 1
                print(f"Skip sentence: Too many calculated paths. At least {len(paths)}.")
                break
            #'''

            word = wd_tag_pair[0]
            current_paths = []
            tags = get_tag_types(word,train_data)

            if ind == 0:
                if not tags:
                    t = rng.choice(all_tags)
                    current_paths.append(([t],1))
                for tag in tags:
                    trans_p = trans_prob_df.at["<S>",tag]
                    emis_p = emis_prob_df.at[word,tag]
                    current_p = trans_p * emis_p
                    if current_p == 0:
                        continue
                    current_tags = [tag]
                    current_paths.append((current_tags,current_p))
            else:
                if not tags:
                    t = rng.choice(all_tags)
                    for tags_l,old_p in paths:
                        current_tags = tags_l + [t]
                        current_paths.append((current_tags,old_p))
                for tag in tags:
                    #Previously, I added emis_p inside the next loop, slowing down the script extremly. It only has to be calculated once per tag.
                    emis_p = emis_prob_df.at[word,tag]
                    for tags_l,old_p in paths:
                        trans_p = trans_prob_df.at[tags_l[-1],tag]
                        current_tags = tags_l + [tag]
                        current_p = old_p * emis_p * trans_p
                        if current_p == 0:
                            continue
                        current_paths.append((current_tags,current_p))
            paths = current_paths
            if not paths:
                break
            max_p = max(paths,key=itemgetter(1))[1]
            paths = [path for path in paths if path[1] == max_p]

        if not paths:
            continue

        for tags_l,old_p in paths:
            trans_p = trans_prob_df.at[tags_l[-1],"<E>"]
            if trans_p == 0:
                continue
            old_p *= trans_p

        max_p = max(paths,key=itemgetter(1))[1]
        paths = [path for path in paths if path[1] == max_p]
        random_ind = np.random.randint(0,len(paths))
        optimal_path = paths[random_ind]
        actual_path = [tag for wd,tag in sent]
        #Check, whether the predictions are correct or not.
        if optimal_path[0] == actual_path:
            sentences_correctly_predicted += 1
            words_correctly_predicted += len(actual_path)
        else:
            for index in range(len(optimal_path[0])):
                if optimal_path[0][index] == actual_path[index]:
                    words_correctly_predicted += 1

    sent_pred_accuracy = sentences_correctly_predicted/len(test_sentences)
    word_pred_accuray = words_correctly_predicted/len(reduce_dim(test_sentences))

    print(f"Skipped sentences due to too many calculated paths: {skipped_sents}/{len(test_sents)}")

    return sent_pred_accuracy,word_pred_accuray


#ACTUAL OPERATIONS BEGIN HERE#
data_path = "../data/"
tagged_words_file = "urum_tagged_words.txt"

#Get the word-tag-pairs from the respective file as tuples inside a list.
tagged_words_l = get_csv_as_list(data_path+tagged_words_file,";")
print(f"Loading data: {tagged_words_file}")

#Put word-tag-pairs into lists representing sentences and put them in one list.
tagged_sentences = get_sent_from_wd_tag_pairs(tagged_words_l,"<E>","<E>")
print(f"Creating sentences.")

sent_acc = 0
word_acc = 0
phases = 20

for i in range(phases):
    print(f"Phase: {i+1}/{phases}")

    #Create the training and test data randomly (optionally with a seed) by default in the ration 8:2 (train:test) from the list of tagged sentences.
    train_sentences,test_sentences = split_train_test(tagged_sentences)#,seed=469283984701)
    #Default seed I chose for comparing results: 469283984701

    #Apply the hmm algorithm and save the results.
    sent_pred,word_pred = viterbi_hmm_algorithm(train_sentences,test_sentences)
    sent_acc += sent_pred
    word_acc += word_pred

print(f"\nSentence Predition Accuracy: {round((sent_acc/phases)*100,2)}%.")
print(f"Word Prediction Accuracy: {round((word_acc/phases)*100,2)}%.")
