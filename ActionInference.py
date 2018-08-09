# This Python file uses the following encoding: utf-8
import random
import QA.Data as Data
from QA.TFIDF import TFIDF
from QA.CNN_train import  CNN
from Chit_Chat.seq2seq import seq2seq

class ActionInference():
    def __init__(self):
        random.seed(12345)

        # QA
        # Read Data
        questions, pred_questions, answers, pred_answers = Data.read_pred_data("data/pred_jd_chat.csv")
        # Build word --> sentence dictionary
        word_sentence_dict = Data.generate_word_sentence_dict(pred_questions)

        self.tfidf = TFIDF(questions, pred_questions, answers, pred_answers, word_sentence_dict)
        self.cnn = CNN(questions, pred_questions, answers, pred_answers, word_sentence_dict, isTrain=False)

        # ---------------

        # Chit chat
        self.seq2seq = seq2seq()

        # ---------------



    def make_action(self, state, action):

        if action == 0:
            top_k = 3
            tfidf_response_id, tfidf_response = self.tfidf.ask_response(state, top_k=top_k)
            # if no answers
            if len(tfidf_response) == 0:
                return "嗯".decode("utf-8")

            # cnn_response = self.cnn.ask_response(state, top_k, tfidf_response_id)
            # return cnn_response[0]
            return tfidf_response[0]

        if action == 1:
            seq2seq_response = self.seq2seq.ask_question(state)
            return seq2seq_response

        return None

def main():
    action_inference = ActionInference()
    print(action_inference.make_action("客服在么".decode("utf-8"), 1))

if __name__=="__main__":
    main()