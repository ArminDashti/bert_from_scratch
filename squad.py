import sys
import json
#%%       
class SQuAD:
    def __init__(self, filename):
        with open(filename, "r") as f:
            self.data = json.load(f)['data']
      
    def QandA(self):
        QandA = []
        for title in self.data:
            paragraphs = title['paragraphs']
            for paragraph in paragraphs:
                qas = paragraph['qas']
                for qa in qas:
                    question = qa['question'] # string
                    answers = qa['answers'] # list
                    for answer in answers:
                        text_answer = answer['text']
                        QandA.append([question,text_answer])
        return QandA
    
    def WoB(self):
        QandA = self.QandA()
        QandA_tokens = {}
        index_token = 0
        for qa in QandA:
            question = qa[0].split()
            answer = qa[1].split()
            for q_token in question:
                index_token += 1
                QandA_tokens[q_token] = index_token
            for a_token in answer:
                index_token += 1
                QandA_tokens[a_token] = index_token
        QandA_tokens = {key: i+5 for i, key in enumerate(sorted(QandA_tokens.keys()))}
        return QandA_tokens
    
    def QandA_with_token_index(self):
        WoB = self.WoB()
        QandA = self.QandA()
        final_list = []
        for word in QandA:
            question = word[0].split()
            answer = word[1].split()
            final_list.append([[WoB[q] for q in question], [WoB[a] for a in answer]])
        return final_list
    
    def padding(self, QandA, max_len=30, sep=True):
        padding_value = 0
        sep_value = [1]
        final_list = []
        for word in QandA:
            question = word[0]
            answer = word[1]
            if sep is not None:
                total = question + sep_value + answer
            else:
                total = question + answer
            if len(total) <= max_len:
                total += [101] * (max_len - len(total))
                final_list.append(total)
        return final_list
                