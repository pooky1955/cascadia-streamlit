import pandas as pd
from functools import reduce
from typing import List, Dict
import numpy as np
from typing import Dict, List

# importing the models


def word_accumulator_fn(info_dict: Dict[str, List], curr_tok):
    if curr_tok['entity'].startswith('I'):
        if len(info_dict['curr_stack']) == 0:
            return info_dict  # just don't modify it
        info_dict['curr_stack'].append(curr_tok)
    else:  # starts with B
        info_dict['word_list'].append(info_dict['curr_stack'])
        info_dict['curr_stack'] = [curr_tok]
    return info_dict

def condense_stack(stack):
    condensed_word = ''.join([part['word'][2:] if part['word'].startswith(
        "##") else ' ' + part['word'] for part in stack]).lstrip()
    scores = np.array([part['score'] for part in stack])
    return condensed_word, scores

def split_one_stack(stack):
    '''Processes only 1 stack and returns a list of untangled stacks (1 stack = 1 entity)'''
    # split section of indices into partitions
    current_ind = -1000
    partitions = []
    current_partition = []
    for el in stack:
        if el['index'] != current_ind + 1:
            # time for new partition
            if len(current_partition) != 0:
                partitions.append(current_partition)
            current_partition = [el]
        else:
            current_partition.append(el)
        current_ind = el['index']

    if len(current_partition) != 0:
        partitions.append(current_partition)

    return partitions


def split_stacks(stacks):
    '''Processes the stacks to return the correct entity stacks. (Untangles multiple entities in 1 stack to ensure each stack = 1 entity)'''
    # sometimes there will be multiple "entities" grouped in 1 stack but with indices that are not contiguous
    return [stack_splitted for stack in stacks for stack_splitted in split_one_stack(stack)]


def get_tokens_from_ner_specific(raw_outputs, entity_type):
    '''gets proper tokens from nlp pipeline (merges subwords as well)'''
    # output is a list of {entity : str, score : float, index : int, word : str, start : int, end : int}
    # goal: merge consecutive indices into one word
    outputs = filter(lambda output: output['entity'].endswith(
        entity_type), raw_outputs)
    try:
        word_stack = reduce(word_accumulator_fn, outputs,
                            dict(word_list=[], curr_stack=[]))
        all_stacks = word_stack['word_list']
    except Exception as e:
        print("exception:", str(e))
        import ipdb
        ipdb.set_trace()

    if len(word_stack['curr_stack']) > 0:
        all_stacks.append(word_stack['curr_stack'])
    # little assert statement for sanity check
    clean_stacks = split_stacks(all_stacks)
    return list(filter(lambda el: el[0] != '', map(condense_stack, clean_stacks)))


def get_tokens_from_ner(raw_outputs, entity_list=['ADR', 'DRUG']):
    return {entity_type: get_tokens_from_ner_specific(raw_outputs, entity_type) for entity_type in entity_list}

def specific_to_df(raw_els_list,term_col,score_col,term_name):
    if len(raw_els_list) == 0:
        return f"Our model didn't detect any {term_name} in the sentence. (It might be wrong!)"
    els_list = [(term,scores.mean()) for term, scores in raw_els_list if scores.mean() > 0.88]
    if len(els_list) == 0:
        return f"Our model is not sure if there are any {term_name} in the sentence. (It might be wrong!)"
    terms, scores = list(zip(*els_list))
    mean_score = [f"{score * 100:.1f}%" for score in scores]
    df = pd.DataFrame()
    df[term_col] = terms
    df[score_col] = mean_score
    return df


def convert_to_df(raw_outputs):
    cleaned_outputs = get_tokens_from_ner(raw_outputs)
    adrs = cleaned_outputs['ADR']
    drugs = cleaned_outputs['DRUG']
    ade_df = specific_to_df(adrs,"Adverse Drug Event Detected", "Confidence","adverse events")
    drug_df = specific_to_df(drugs,"Drugs Detected","Confidence","drugs")
    return ade_df, drug_df
    


