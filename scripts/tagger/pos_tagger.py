#Script written by Aleksandr Schamberger (GitHub: https://github.com/a-leks-icon) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt Universität zu Berlin in the winter semester 2023/24.

#Script is inspired by the blog post from MyGreatLearning <https://www.mygreatlearning.com/blog/pos-tagging/> (accessed on 2024-01-28).

import pandas as pd
import numpy as np
import math
from operator import itemgetter

def get_csv_as_list(file_path:str,separator:str=";") -> list[tuple[str,str]]:
    '''Imports a .csv-file as a list.

    Imports a .csv-file **file_path** with **separator** as the delimiter and returns it as a list with tuples. Every tuple represents a row/line and every element in a tuple represents a unique value in a column.

    Args:
        file_path: Path to the .csv-file.
        separator: Delimiter used to separate datapoints in the .csv-file.

    Returns:
        A list of tuples.
    '''
    tagged_tokens = pd.read_csv(file_path,sep=separator,header=None,on_bad_lines="warn")
    tagged_tokens = tagged_tokens.dropna()
    tagged_tokens_l = tagged_tokens.values.tolist()
    return [(i[0],i[1]) for i in tagged_tokens_l]

def get_sent_from_wd_tag_pairs(tok_tag_pairs:list,tok_end:str,tag_end:str) -> list[list]:
    '''Create a list containing lists representing sentences.

    Creates lists representing sentences, where each list contains any number of token-tag-pairs taken from a list of tuples **wd_tag_pairs**. Uses tuples with **wd_end** and **tag_end** as their values to encode the end of a sentence and the start of the next sentence. In the end, appends all lists to one list and returns the latter.

    Args:
        tok_tag_pairs: List of tuples with two elements. The first element represents (word or morph) tokens; the second a (POS) tag.
        tok_end: String for identifying the end of a sentence for tokens (the tuple's first element).
        tag_end: String for identifying the end of a sentence for tags (the tuple's second element).

    Returns:
        List with any number of lists containing any number of tuples with two elements.
    '''
    tagged_sents = []
    sent = []
    for tok,tag in tok_tag_pairs:
        if (tok != tok_end) & (tag != tag_end):
            sent.append((tok,tag))
        elif sent:
            tagged_sents.append(sent)
            sent = []
    return tagged_sents

def split_train_test(data:list,size:float=0.8,seed:bool|int|float=False) -> tuple[list,list]:
    '''Splits **data** into training and test data.

    Splits a list with lists representing sentences **data** into two separate lists by pseudo randomly shuffling **data** with a pseudorandom **seed**. The size of the training data equals the length of **data* multiplied by **size**. The test data amounts to the remaining data.

    Args:
        data: List containing lists representing sentences. Each list contains tuples representing token-tag-pairs.
        size: Float representing the size of the training data.
        seed: Number to initialize the pseudorandom number generator used to randomly split the data into training and test data. Used to control how the data is being split. Using the same number results in the same outcome for the training and test data. By default, **seed** is False indicating that no random seed is given.

    Returns:
        Tuple with two lists containing the training and test data respectively.
    '''
    if isinstance(seed,bool):
        rng = np.random.default_rng()
    elif isinstance(seed,(int,float)):
        rng = np.random.default_rng(seed)
    split_num = math.ceil(len(data)*size)
    random_data = data
    rng.shuffle(random_data)
    train = random_data[:split_num]
    test = random_data[split_num:]
    return (train,test)

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

def get_emission_probability(training_data:list) -> pd.DataFrame:
    '''Creates a pandas data frame with emission probabilities.

    Returns a pandas data frame containing the emission probability of every unique token-tag-pair from a given list of token-tag-pairs **training_data**. In the final data frame, rows represent (word or morph) types and columns represent unique (POS) tags.

    Args:
        training_data: List with tuples representing token-tag-pairs to train a model.

    Returns:
        Pandas data frame with emission probabilities. Rows represent (word or morph) types and columns represent unique (POS) tags.
    '''

    def get_emission_probability_base(type:str,tag:str,training_data:list) -> float:
        '''Returns the emission probability.

        Returns a float representing the emission probability of a (word or morph) **type** given a (POS) **tag** from a list of token-tag-pairs **training_data**.

        Args:
            type: String representing a (word or morph) type.
            tag: String representing a unique (POS) tag.
            training_data: List with tuples representing token-tag-pairs to train a model.

        Returns:
            Float representing the emission probability.
        '''
        #P(word|tag) = (P(word) ∩ P(tag))/P(tag)
        #P(word|tag) = C(<tag,word>)/C(<tag,X>)
        pairs_with_tag = [(tp,t) for tp,t in training_data if t == tag]
        tag_count = len(pairs_with_tag)
        type_tag_count = len([tp for tp,t in pairs_with_tag if tp == type])
        return type_tag_count/tag_count

    types = tuple(set([token for token,tag in training_data]))
    unique_tags = list(set([tag for token,tag in training_data]))
    emis_prob_array  = np.zeros((len(types),len(unique_tags)))
    for type_ind,type in enumerate(types):
        relevant_unique_tags = tuple(set([tag for token,tag in training_data if token == type]))
        for tag in relevant_unique_tags:
            tag_ind = unique_tags.index(tag)
            emis_prob_array[type_ind,tag_ind] = get_emission_probability_base(type,tag,training_data)
    return pd.DataFrame(emis_prob_array,index=types,columns=unique_tags)

def get_transition_probability(training_sentences:list) -> pd.DataFrame:
    '''Creates a pandas data frame with transition probabilities.

    Return a pandas data frame containing the transition probability of every pair of unique (POS) tags including the pseudo tags *<S>* and *<E>* given a list of **training_sentences**. In the final data frame, the rows represent the first and the columns the second tag.

    Args:
        training_sentences: List containing lists representing sentences. Each list contains tuples representing token-tag-pairs used to train a model.

    Returns:
        Pandas data frame with transition probabilities. Rows and columns represent unique (POS) tags.
    '''

    def get_transition_probability_base(tag1:str,tag2:str,training_data:list) -> float:
        '''Returns the transition probability.

        Returns a float representing the transition probability of a tag **tag2** given its previous tag **tag1** from a list of token-tag-pairs **training_data**.
        
        Args:
            tag1: String representing the preceding tag.
            tag2: String representing the following tag.
            training_data:  List with tuples representing token-tag-pairs to train a model.

        Returns:
            Float representing the transition probability.
        '''
        #P(tag2|tag1) = C(<tag1,tag2>)/C(<tag1,X>)
        tags = [tag for token,tag in training_data]
        count_tag1_tag2 = 0
        for ind in range(len(tags)-1):
            if (tags[ind] == tag1) & (tags[ind+1] == tag2):
                count_tag1_tag2 += 1
        count_tag1_x = len([tag for tag in tags if tag == tag1])
        return count_tag1_tag2/count_tag1_x

    new_sentences = [sent.copy() for sent in training_sentences]
    for sent in new_sentences:
        sent.insert(0,("<S>","<S>"))
        sent.append(("<E>","<E>"))
    new_data = reduce_dim(new_sentences)
    unique_tags = list(set([tag for token,tag in new_data]))
    trans_prob_array = np.zeros((len(unique_tags),len(unique_tags)))
    for ind1,t1 in enumerate(unique_tags):
        for ind2,t2 in enumerate(unique_tags):
            if t1 == "<E>":
                trans_prob_array[ind1,ind2] = 0
            elif t2 == "<S>":
                trans_prob_array[ind1,ind2] = 0
            else:
                trans_p = get_transition_probability_base(t1,t2,new_data)
                trans_prob_array[ind1,ind2] = trans_p
    return pd.DataFrame(trans_prob_array,columns=unique_tags,index=unique_tags)

def get_most_frequent_tag(training_data:list) -> str:
    '''Get the most frequently occuring tag in **training_data**.

    Args:
        training_data: List with tuples representing token-tag-pairs to train a model.

    Returns:
        String representing the most frequently occuring tag.
    '''
    unique_tags = list(set([tag for token,tag in training_data]))
    tag_freq = []
    for tag in unique_tags:
        freq = len([t for tok,t in training_data if t == tag])
        tag_freq.append((tag,freq))
    most_frequent_tag = max(tag_freq,key=itemgetter(1))[0]
    return most_frequent_tag

def hmm_and_viterbi_algorithm(train_sentences:list,test_sentences:list) -> tuple[float,float]:
    '''Initializes, trains and tests a tagger.

    Creates and trains an HMM based on **train_sentences**. The model is used as a tagger, which is tested on **test_sentences** using the Viterbi algorithm.

    Args:
        train_sentences: List containing lists representing sentences. Each list contains tuples representing token-tag-pairs. The whole list represents the training data.
        test_sentences: List containing lists representing sentences. Each list contains tuples representing token-tag-pairs. The whole list represents the test data.

    Returns:
        Tuple with three floats. The first float represents the tagger's accuracy correctly predicting a sequence of tags (sentence). The second float represents the tagger's accuracy correctly predicting single tags (token). The third float represents the average number of tokens appearing in the test but not in the training data.
    '''

    def get_optimal_seq(seqs_probs:dict,seqs_key:str="seqs",probs_key:str="probs") -> tuple:
        '''Gets the best sequence and probability.

        Returns the optimal sequence and its probability as a tuple based on a dict **seqs_probs** containing i) a list of sequences of (POS) tags as a value for the key **seqs_key** and ii) a list of associated computed probabilities as a value for the key **probs_key**.

        Args:
            seqs_probs: Dictionary with two keys **seqs_key** and **probs_key**. The value of the first key is a list of sequences of (POS) tags (list of lists containing strings). The value of the second key is a list of probabilities (list of floats) belonging to the sequences in **seqs_key**.
            seqs_key: String to access the list of sequences of POS tags (list of lists containing strings).
            probs_key: String to access the list of probabilities (list of floats).

        Returns:
            Tuple with two elements. The first element is a list of strings representing the optimal sequence of (POS) tags. The second element is a float representing the probability belonging to the optimal sequence.
        '''
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
    #Train the hidden Markov model: Create the data frames for the emission and transition probabilities.
    trans_p_df = get_transition_probability(train_sentences)
    train_data = reduce_dim(train_sentences)
    emis_p_df = get_emission_probability(train_data)

    #Create a vocabulary and list of unique POS tags given the training data.
    voc = list(emis_p_df.index)
    tags = list(emis_p_df.columns)

    print(f"Testing.")
    #Counting the number of correctly predicted i) sentences and ii) tokens.
    sentences_correctly_predicted = 0
    tokens_correctly_predicted = 0
    #Counting the number of out of vocabulary tokens.
    oov = 0
    most_frequent_tag = get_most_frequent_tag(train_data)

    #Iterate through all test sentences.
    for sent in test_sentences:
        #Create a list for sequences and their associated probabilities.
        seqs_probs = {"seqs": [["<S>"]],
                      "probs": [1]}
        #Iterate through pairs of tokens and POS tags in a sentence except for the first and last.
        for n in range(len(sent)):
            #Get the token.
            token = sent[n][0]
            #Prepare lists for collecting the new best sequences and probabilites.
            new_seqs = []
            new_probs = []
            #If the token does not occur in the training data: Iterate through every sequence and add the most frequently occuring POS tag in the training data to the sequence.
            if not token in voc:
                oov += 1
                best_seq,best_p = get_optimal_seq(seqs_probs)
                new_seqs.append(best_seq + [most_frequent_tag])
                new_probs.append(best_p)
            else:
                #If the token occurs in the training data: Iterate through all relevant tags.
                for tag in tags:
                    #Initialize variables to keep track of the best sequence a POS tag occurs in (backtrack).
                    best_seq_ind = None
                    best_p = 0
                    #Calculate the emission probability (doing it later slows down the code because of repeated computation)
                    emis_p = emis_p_df.at[token,tag]
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
                            best_seq_ind = i
                    #Get the best sequence, probability and tag, and add it to the list of new sequences and probabilities, if it is positive.
                    if best_p > 0:
                        new_seqs.append(seqs_probs["seqs"][best_seq_ind] + [tag])
                        new_probs.append(best_p)
            #Before continuing with the next token, update the lists of sequences and probabilities for the next iteration step.
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
        #Test, whether the predicted sequence of POS tags is correct or not. If so, increase the number of correctly predicted sentences and/or tokens.
        actual_seq = [tag for token,tag in sent]
        if optimal_seq[1:] == actual_seq:
            sentences_correctly_predicted += 1
            tokens_correctly_predicted += len(sent)
        else:
            for n in range(len(optimal_seq[1:])):
                if optimal_seq[n+1] == actual_seq[n]:
                    tokens_correctly_predicted += 1
    #Calculate the amount of correctly predicted sentences and (word or morph) tokens normalized by the number of predictable elements (sentences and/or tokens), and return both values.
    sent_pred_accuracy = sentences_correctly_predicted/len(test_sentences)
    tok_pred_accuracy = tokens_correctly_predicted/len(reduce_dim(test_sentences))
    oov = oov/len(reduce_dim(test_sentences))
    return sent_pred_accuracy,tok_pred_accuracy,oov

#START OPERATIONS#

#Path to preprocessed data.
data_path = "../../data/preprocessed/"
#File to be processed.
dataset = "urum_words.txt"

#Get token-tag-pairs from the respective file as tuples inside a list.
print(f"Loading dataset: {dataset}")
tagged_tokens = get_csv_as_list(data_path+dataset,";")

#First, append token-tag-pairs to lists, which represent sentences. Second, append every list to one main list.
tagged_sentences = get_sent_from_wd_tag_pairs(tagged_tokens,"<E>","<E>")

#Training parameters: Number of training and test iterations.
epochs = 100

#The statistical model used to evaluate the performace of the tagger:
#(1) "values": Accuracy for correctly tagged i) sentences and ii) tokens,
#or the number of out-of-vocabulary tokens.
#number of tokens in the test data.
#(2): "mean": Mean of all prediction accuracies.
#(3): "sd": Standard deviation of the mean.
stats = {
    "sentence": {"values":[],
                 "mean":0,
                 "sd":0,
                 "max":0,
                 "min":0},
    "token": {"values":[],
              "mean":0,
              "sd":0,
              "max":0,
              "min":0},
    "oov": {"values":[],
            "mean":0,
            "sd":0,
            "max":0,
            "min":0}
}

for i in range(epochs):
    print(f"Epoch: {i+1}/{epochs}")

    #Create the training and test data randomly (optionally with a seed) by default in the ration 80:20 (train:test) from the list of tagged sentences.
    train_sentences,test_sentences = split_train_test(tagged_sentences,seed=469283984701)
    #Default seed I chose for comparing results: 469283984701

    #Train an HMM and test it using the Viterbi algorithm. Save the test results.
    sent_pred,tok_pred,oov = hmm_and_viterbi_algorithm(train_sentences,test_sentences)
    stats["sentence"]["values"].append(sent_pred)
    stats["token"]["values"].append(tok_pred)
    stats["oov"]["values"].append(oov)

#Calculate statistics.
for key,val in stats.items():
    values = val["values"]
    #Mean.
    mean = sum(values)/len(values)
    squared = [(n - mean) ** 2 for n in values]
    ss = sum(squared)
    variance = ss/(len(values)-1)
    #Standard deviation.
    sd = variance ** (1/2)
    #Highest and lowest value.
    val["max"] = max(values)
    val["min"] = min(values)
    #Save statistics.
    val["mean"] = mean
    val["sd"] = sd

#Print the results to the terminal.
print(f"\nSentence predition accuracy: {round(stats['sentence']['mean']*100,2)}%. SD: {round(stats['sentence']['sd']*100,2)}. Max: {round(stats['sentence']['max']*100,2)}%. Min {round(stats['sentence']['min']*100,2)}%.")
print(f"Token prediction accuracy: {round(stats['token']['mean']*100,2)}%. SD: {round(stats['token']['sd']*100,2)}. Max: {round(stats['token']['max']*100,2)}%. Min {round(stats['token']['min']*100,2)}%.")
print(f"Out-of-vocabulary tokens per epoch: {round(stats['oov']['mean']*100,2)}%. SD: {round(stats['oov']['sd']*100,2)}. Max: {round(stats['oov']['max']*100,2)}%. Min {round(stats['oov']['min']*100,2)}%.")
