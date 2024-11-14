#Script written by Aleksandr Schamberger (GitHub: https://github.com/a-leks-icon) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt-University Berlin in the winter semester 2023/24.

#Script is partly based on the blog post from MyGreatLearning <https://www.mygreatlearning.com/blog/pos-tagging/> (link lastly opened on 2024-01-28).

#Created: 2024-09-20
#Latest Version: 2024-09-21

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

def get_relevant_tags(word:str,data):
    '''Returns a list with relevant POS tags for *word* from a list of tuples of word-tag pairs *data*.'''
    unique_tags = set()
    for wd,tag in data:
        if wd == word:
            unique_tags.add(tag)
    return list(unique_tags)

def get_most_common_tag(training_data):
    '''Returns given a list of word-tag-pairs *data* the most frequently occuring POS tag in the *training_data*.'''
    unique_tags = list(set([tag for wd,tag in training_data]))
    tag_freq = []
    for tag in unique_tags:
        freq = len([t for wd,t in training_data if t == tag])
        tag_freq.append((tag,freq))
    most_common_tag = max(tag_freq,key=itemgetter(1))[0]
    return most_common_tag

def get_tags(data):
    '''Returns a list of unique POS tags given a list of word-tag-pairs *data*.'''
    unique_tags = list(set([tag for wd,tag in data]))
    return unique_tags

def hmm_and_viterbi_algorithm(train_sentences,test_sentences):
    '''Creates and trains an HMM based on *train_sentences*, and tests the model on *test_sentences* using the Viterbi algorithm. Returns a tuple with two values: i) The accuracy predicting sequences of POS tags for sentences correctly, and ii) the accuracy prediciting a word's correct POS tag.'''

    def get_optimal_seq(seqs_probs:dict,seqs_key="seqs",probs_key="probs"):
        '''Returns the optimal sequence and its probability as a tuple based on a dict *seqs_probs* containing a list of sequences of POS tags with key *seqs_key* and a list of associated computed probabilities with key *probs_key*.'''
        optimal_seq = None
        best_p = 0
        for n in range(len(seqs_probs[seqs_key])):
            seq_p = seqs_probs[probs_key][n]
            seq = seqs_probs[seqs_key][n]
            if seq_p > best_p:
                best_p = seq_p
                optimal_seq = seq
        return optimal_seq, best_p

    print(f"Training.")
    trans_p_df = get_transition_probability(train_sentences)
    train_data = reduce_dim(train_sentences)
    emis_p_df = get_emission_probability(train_data)

    print(f"Predicting.")
    #Iterating over every tag, even if a word does not appear with that tag at all, seems to be slower than first getting only the relevant tags and only then iterating through those.
    sentences_correctly_predicted = 0
    words_correctly_predicted = 0
    most_common_tag = get_most_common_tag(train_data)

    #Iterate through all test sentences.
    for sent in test_sentences:
        #Create a list for sequences and their associated probabilities.
        seqs_probs = {"seqs": [["<S>"]],
                      "probs": [1]}
        #Iterate through pairs of words and POS tags in a sentence except for the first and last.
        for n in range(len(sent)):
            #Get the word.
            word = sent[n][0]
            #Get all POS tags a word occurs as.
            tags = get_relevant_tags(word,train_data)
            #Prepare lists for collecting the new best sequences and probabilites.
            new_seqs = []
            new_probs = []
            #If the word does not occur in the training data: Iterate through every sequence and add the most frequently occuring POS tag in the training data to the sequence.
            if not tags:
                best_seq,best_p = get_optimal_seq(seqs_probs)
                new_seqs.append(best_seq + [most_common_tag])
                new_probs.append(best_p)
            #If the word occurs in the training data: Iterate through all relevant tags.
            for tag in tags:
                #Initialize variables to keep track of the best sequence a POS tag occurs in (backtrack).
                best_tag = None
                best_seq_ind = None
                best_p = 0
                #Calculate the emission probability (doing it later slows down the code because of repeated computation)
                emis_p = emis_p_df.at[word,tag]
                #Iterate thorugh every sequence and probability collected so far.
                for i in range(len(seqs_probs["seqs"])):
                    seq = seqs_probs["seqs"][i]
                    prev_tag = seq[-1]
                    #Compute the transition probability, get the previous probability and compute the new probability of the sequence.
                    prev_p = seqs_probs["probs"][i]
                    trans_p = trans_p_df.at[prev_tag,tag]
                    seq_p = prev_p * emis_p * trans_p
                    #Check, whether the new sequence probability is the best so far.
                    if seq_p > best_p:
                        best_p = seq_p
                        best_tag = tag
                        best_seq_ind = i
                #Get the best sequence, probability and tag, and add it to the list of new sequences and probabilities, if it is positive.
                if best_p > 0:
                    new_seqs.append(seqs_probs["seqs"][best_seq_ind] + [best_tag])
                    new_probs.append(best_p)
            #Before continuing with the next word, update the lists of sequences and probabilities for the next iteration step.
            seqs_probs["seqs"] = new_seqs
            seqs_probs["probs"] = new_probs
            #Check, whether sequences with values greater than 0 are available. If not, break.
            if not seqs_probs["seqs"]:
                break
        #Create empty lists for collecting sequences with probabilities greater than 0.
        new_seqs = []
        new_probs = []
        #Include the end tag <E> into the sequences and then pick out the sequence with the highest computed probability.
        for n in range(len(seqs_probs["seqs"])):
            seq = seqs_probs["seqs"][n]
            prev_tag = seq[-1]
            trans_p = trans_p_df.at[prev_tag,"<E>"]
            if trans_p > 0:
                prev_p = seqs_probs["probs"][n]
                new_seqs.append(seq)
                new_probs.append(prev_p * trans_p)
        #Update the list of sequences and probabilities a last time.
        seqs_probs["seqs"] = new_seqs
        seqs_probs["probs"] = new_probs
        #Check, whether sequences with probabilities greater than 0 remain. If not, continue with the next test sentence.
        if not seqs_probs["probs"]:
            continue
        #Get the sequence with the highest computed probability.
        optimal_seq = get_optimal_seq(seqs_probs)[0]
        #Test, whether the predicted sequence of POS tags is correct or not. If so, increase the number of correctly predicted sentences and/or words
        actual_seq = [tag for wd,tag in sent]
        if optimal_seq[1:] == actual_seq:
            sentences_correctly_predicted += 1
            words_correctly_predicted += len(sent)
        else:
            for n in range(len(optimal_seq[1:])):
                if optimal_seq[n+1] == actual_seq[n]:
                    words_correctly_predicted += 1
    #Calculate the amount of correctly predicted sentences and words normalized by the number of predictable elements (sentences and/or words), and return both values.
    sent_pred_accuracy = sentences_correctly_predicted/len(test_sentences)
    word_pred_accuracy = words_correctly_predicted/len(reduce_dim(test_sentences))
    return sent_pred_accuracy,word_pred_accuracy

#START OPERATIONS#
data_path = "../data/"
tagged_words_file = "urum_tagged_words.txt"

#Get the word-tag-pairs from the respective file as tuples inside a list.
tagged_words_l = get_csv_as_list(data_path+tagged_words_file,";")
print(f"Loading data: {tagged_words_file}")

#Put word-tag-pairs into lists representing sentences and put them in one list.
tagged_sentences = get_sent_from_wd_tag_pairs(tagged_words_l,"<E>","<E>")
print(f"Creating sentences.")

#Training parameters.

#Accuracy for correctly tagged i) sentences and ii) words.
sent_acc = 0
word_acc = 0

#Number of training and test iterations.
epochs = 20

for i in range(epochs):
    print(f"Phase: {i+1}/{epochs}")

    #Create the training and test data randomly (optionally with a seed) by default in the ration 8:2 (train:test) from the list of tagged sentences.
    train_sentences,test_sentences = split_train_test(tagged_sentences,seed=469283984701)
    #Default seed I chose for comparing results: 469283984701

    #Train an HMM and test it using the Viterbi algorithm. Save the test results.
    sent_pred,word_pred = hmm_and_viterbi_algorithm(train_sentences,test_sentences)
    sent_acc += sent_pred
    word_acc += word_pred

print(f"\nSentence Predition Accuracy: {round((sent_acc/epochs)*100,2)}%.")
print(f"Word Prediction Accuracy: {round((word_acc/epochs)*100,2)}%.")
