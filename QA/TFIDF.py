# This Python file uses the following encoding: utf-8
import Data
import heapq
import numpy as np
import random
import csv


class TFIDF:
    def __init__(self, questions=None, pred_questions=None, answers=None, pred_answers=None,
                 word_sentence_dict=None):
        # Read Preprocessed Data
        if questions == None:
            self.questions, self.pred_questions, self.answers, self.pred_answers = Data.read_pred_data(
                "Data/pred_QA-pair.csv")
        else:
            self.questions, self.pred_questions, self.answers, self.pred_answers = questions, pred_questions, answers, pred_answers

        if word_sentence_dict == None:
            # Build word --> sentence dictionary
            self.word_sentence_dict = Data.generate_word_sentence_dict(self.pred_questions)
        else:
            self.word_sentence_dict = word_sentence_dict

        # Calculate TF-IDF
        self.idf_dict = generate_idf_dict(self.pred_questions)

        self.tf_idf_pred_questions = generate_tf_idf_list(self.pred_questions, self.idf_dict)

    def ask_response(self, question, top_k):
        """
        :param question: input a question
        :return: top k id and response
        """
        pred_q = Data.preprocessing([question])
        tf_idf_pred_q = generate_tf_idf_list(pred_q, self.idf_dict)

        top = []

        # Generate sentence id set which include at least one same word
        sentence_id_set = set()
        for j in range(len(pred_q[0])):
            if pred_q[0][j] in self.word_sentence_dict:
                sentence_id_set.update(self.word_sentence_dict[pred_q[0][j]])

        # Generate cosine similarity score
        for j in sentence_id_set:
            score = cosine_similarity(tf_idf_pred_q[0], self.tf_idf_pred_questions[j])
            heapq.heappush(top, (-score, str(j)))

        # print("Question: %s" % question)

        response = []
        response_id = []

        # Generate Top K
        for j in range(min(top_k, len(top))):
            item = int(heapq.heappop(top)[1])
            # print("Similar %d: %s" % (j + 1, self.questions[item]))
            # print("TFIDF Response %d: %s" % (j + 1, self.answers[item]))
            response_id.append(item)
            response.append(self.answers[item])

        # print("")

        return response_id, response


def generate_idf_dict(word_list):
    """
    Generate word dictionary based on train data
    """
    dict = {}
    for i in range(len(word_list)):
        flag = set()
        for j in range(len(word_list[i])):
            if word_list[i][j] in flag: continue
            if word_list[i][j] not in dict:
                dict[word_list[i][j]] = 1
            else:
                dict[word_list[i][j]] += 1
            flag.add(word_list[i][j])

    return dict


def generate_tf_idf_list(sentences, idf_dict):
    """
    Generate tf-idf for each word in each sentence
    """
    tf_idf = []
    for sentence in sentences:
        dict = {}
        # Get term frequency
        for word in sentence:
            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1

        # Calculate TF-IDF
        for word in dict:
            if word in idf_dict:
                dict[word] = (1 + np.log(dict[word])) * np.log(len(idf_dict) / (idf_dict[word] + 0.0))
            else:
                dict[word] = 0
        tf_idf.append(dict)

    return tf_idf


def cosine_similarity(dict_x, dict_y):
    """
    Calculate Cosine similarity
    """

    def multiply(dict_u, dict_v):
        """
        Multiply dictionaries
        """
        mul = 0.0
        for word in dict_u:
            if word in dict_v:
                mul += dict_u[word] * dict_v[word]
        return mul

    if len(dict_x) == 0 or len(dict_y) == 0: return 0.0
    return multiply(dict_x, dict_y) / (np.sqrt(multiply(dict_x, dict_x)) * np.sqrt(multiply(dict_y, dict_y)))


def file_output(input_file_name, output_file_name,top_k=3):
    """
    tfidf file output
    """
    # Read Preprocessed Data
    questions, pred_questions, answers, pred_answers = Data.read_pred_data(input_file_name)

    pair = list(zip(questions, pred_questions, answers, pred_answers))
    random.shuffle(pair)
    questions, pred_questions, answers, pred_answers = zip(*pair)

    # Split Data
    split_ratio = 0.7
    split_len = int(len(questions) * split_ratio)
    train_questions = questions[:split_len]
    train_pred_questions = pred_questions[:split_len]
    train_answers = answers[:split_len]
    train_pred_answers = pred_answers[:split_len]
    test_questions = questions[split_len:]
    test_pred_questions = pred_questions[split_len:]
    test_answers = answers[split_len:]
    test_pred_answers = pred_answers[split_len:]

    # Calculate TF-IDF
    idf_dict = generate_idf_dict(train_pred_questions)
    print("Word Num is %d" % len(idf_dict))

    tf_idf_train_pred_questions = generate_tf_idf_list(train_pred_questions, idf_dict)
    tf_idf_test_pred_questions = generate_tf_idf_list(test_pred_questions, idf_dict)

    # Build word --> sentence dictionary
    word_sentence_dict = Data.generate_word_sentence_dict(train_pred_questions)
    # print(word_sentence_dict)

    # Txt output
    # # Choose the Top K similar ones
    if output_file_name.split(".")[-1] == "txt":
        output = open(output_file_name, 'w')
        for i in range(len(tf_idf_test_pred_questions)):
            top = []

            # Generate sentence id set which include at least one same word
            sentence_id_set = set()
            for j in range(len(test_pred_questions[i])):
                if test_pred_questions[i][j] in word_sentence_dict:
                    sentence_id_set.update(word_sentence_dict[test_pred_questions[i][j]])
                    # print test_pred_questions[i][j],
            # print(len(sentence_id_set))

            # Generate cosine similarity score
            for j in sentence_id_set:
                score = cosine_similarity(tf_idf_test_pred_questions[i], tf_idf_train_pred_questions[j])
                heapq.heappush(top, (-score, str(j)))

            output.write("Question: " + test_questions[i].encode("utf-8") + "\n")
            # output.write("Ground Truth: " + test_answers[i].encode("utf-8") + "\n")

            # Generate Top K
            for j in range(min(top_k, len(top))):
                item = int(heapq.heappop(top)[1])
                # output.write("Our similar " + str(j + 1) + ": " + train_questions[item].encode("utf-8") + "\n")
                output.write("Our reply " + str(j + 1) + ": " + train_answers[item].encode("utf-8") + "\n")
            output.write("\n")
        output.close()

    # CSV output
    # Choose the Top K similar ones
    if  output_file_name.split(".")[-1] == "csv":
        with open(output_file_name, 'w', ) as csvfile:
            fieldnames = ['Question']
            fieldnames.extend(["Reply " + str(i + 1) for i in range(top_k)])
            fieldnames.append("Score")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(tf_idf_test_pred_questions)):
                top = []
                dict = {"Score": ""}
                # Generate sentence id set which include at least one same word
                sentence_id_set = set()
                for j in range(len(test_pred_questions[i])):
                    if test_pred_questions[i][j] in word_sentence_dict:
                        sentence_id_set.update(word_sentence_dict[test_pred_questions[i][j]])

                # Generate cosine similarity score
                for j in sentence_id_set:
                    score = cosine_similarity(tf_idf_test_pred_questions[i], tf_idf_train_pred_questions[j])
                    heapq.heappush(top, (-score, str(j)))

                dict["Question"] = test_questions[i].encode("utf-8")

                # Generate Top K
                for j in range(min(top_k, len(top))):
                    item = int(heapq.heappop(top)[1])
                    dict["Reply " + str(j + 1)] = train_answers[item].encode("utf-8")
                writer.writerow(dict)


def main():
    file_output("Data/pred_QA-pair.csv", "Data/TFIDF.csv", 3)
    # tfidf = TFIDF()
    # tfidf.ask_response("安装费用", top_k= 3)
    # tfidf.ask_response("什么时候有货", top_k= 3)


if __name__ == "__main__":
    main()
