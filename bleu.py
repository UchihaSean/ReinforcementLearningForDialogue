# -*- coding: UTF-8 -*-
import csv
import numpy as np

def evaluation_bleu(eval_sentence, base_sentence, n_gram=2):
    """
    BLEU evaluation with n-gram
    """

    def generate_n_gram_set(sentence, n):
        """
        Generate word set based on n gram
        """
        n_gram_set = set()
        for i in range(len(sentence) - n + 1):
            word = ""
            for j in range(n):
                word += sentence[i + j]
            n_gram_set.add(word)
        return n_gram_set

    if n_gram > len(eval_sentence) or n_gram > len(base_sentence): return 0.0
    base_n_gram_set = generate_n_gram_set(base_sentence, n_gram)
    eval_n_gram_set = generate_n_gram_set(eval_sentence, n_gram)
    # print(list(base_n_gram_set)[0])
    # print(list(eval_n_gram_set)[1])

    bleu = 0.0
    for word in eval_n_gram_set:
        if word in base_n_gram_set:
            bleu += 1
    return bleu / len(eval_n_gram_set)

def file_evaluation_bleu(file_name):
    bleu_set = []
    with open(file_name, 'r') as csvfile:
        file_info = csv.reader(csvfile)
        for i, line in enumerate(file_info):
            if i==0: continue
            bleu = evaluation_bleu(line[2].decode("utf-8"), line[1].decode("utf-8"))
            # print(line[1])
            # print(line[2])
            bleu_set.append(bleu)

    print(np.mean(bleu_set))



def main():
    # print(evaluation_bleu("你们好".decode("utf-8"), "你们".decode("utf-8")))
    file_evaluation_bleu("data/rl_output.csv")

if __name__ == '__main__':
    main()