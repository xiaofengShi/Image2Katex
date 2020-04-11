'''
File: text.py
Project: evaluate
File Created: Wednesday, 28th November 2018 4:14:46 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Saturday, 22nd December 2018 4:00:46 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
 2018.06 - 2018 Latex Math, Latex Math
'''

import os
import sys

import distance
import nltk
import numpy as np


def load_formulas(filename):
    formulas = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()

    print("Loaded {} formulas from {}".format(len(formulas), filename))
    return formulas


def cal_score(path_ref, path_hyp):
    """Loads result from file and score it

    Args:
        path_ref: (string) formulas of reference
        path_hyp: (string) formulas of prediction.

    Returns:
        scores: (dict)

    """
    # load formulas
    formulas_ref = load_formulas(path_ref)
    formulas_hyp = load_formulas(path_hyp)

    assert len(formulas_ref) == len(formulas_hyp)

    # tokenize
    refs = [ref.split(' ') for _, ref in formulas_ref.items()]
    hyps = [hyp.split(' ') for _, hyp in formulas_hyp.items()]

    # score
    return {
        "BLEU-4": bleu_score(refs, hyps)*100,
        "EM": exact_match_score(refs, hyps)*100,
        "Edit": edit_distance(refs, hyps)*100
    }


def exact_match_score(references, hypotheses):
    """Computes exact match scores.

    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)

    Returns:
        exact_match: (float) 1 is perfect
    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """Computes bleu score.

    Args:
        references: list of list (one hypothesis)
        hypotheses: list of list (one hypothesis)

    Returns:
        BLEU-4 score: (float)
    """
    references = [[ref] for ref in references]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses,
                                                   weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4


def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.

    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)

    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)

    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot


def truncate_end(list_of_ids, id_end):
    """Removes the end of the list starting from the first id_end token"""
    list_trunc = []
    for idx in list_of_ids:
        if idx == id_end:
            break
        else:
            list_trunc.append(idx)

    return list_trunc


def write_answers(references, hypotheses, rev_vocab, dir_name, id_end, batch_name=None):
    """Writes text answers in files.
    One file for the reference, one file for each hypotheses
    Args:
        references: list of list         (one reference)
        hypotheses: list of list of list (multiple hypotheses)
            hypotheses[0] is a list of all the first hypothesis for all the
            dataset
        rev_vocab: (dict) rev_vocab[idx] = word
        dir_name: (string) path where to write results
        id_end: (int) special id of token that corresponds to the END of
            sentence
    Returns:
        file_names: list of the created files
    """
    def ids_to_str(ids):
        ids = truncate_end(ids, id_end)
        s = [rev_vocab[idx] for idx in ids]
        return u" ".join(s)

    def write_file(file_name, list_of_list):
        with open(file_name, "w") as f:
            for l in list_of_list:
                f.write(ids_to_str(l) + "\n")

    def write_file_name(file_name, list_of_list):
        with open(file_name, 'w') as f:
            f.write('\n'.join(list_of_list))

    file_names = [os.path.join(dir_name, "label.txt")]
    write_file(os.path.join(dir_name, "label.txt"), references)  # one file for the ref
    if batch_name is not None:
        write_file_name(os.path.join(dir_name, "names.txt"), batch_name)  # one file for the ref

    for i in range(len(hypotheses)):                            # one file per hypo
        assert len(references) == len(hypotheses[i])
        write_file(os.path.join(dir_name, "predict_{}.txt".format(i)), hypotheses[i])
        file_names.append(os.path.join(dir_name, "predict_{}.txt".format(i)))

    return file_names
