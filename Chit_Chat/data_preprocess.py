

def preprocess(input_file_name, output_file_name):
    """
    generate jd chat file for qa
    """
    input_file = open(input_file_name)
    output_file = open(output_file_name, 'w')
    chat_id = ""
    question = ""
    answer = ""
    isServe = -1
    j = 0
    session = 0
    for i, line in enumerate(input_file.readlines()):

        if i == 0: continue

        # if i > 100: break

        lines = line.strip().split("	")
        if chat_id == "":
            chat_id = lines[0]
            session+=1
        else:
            if chat_id != lines[0]:
                if question!="" and answer!="":
                    output_file.write(str(j * 2) + " +++$+++ " + str(session) + " +++$+++ " + question + "\n")
                    output_file.write(str(j * 2 + 1) + " +++$+++ " + str(session) + " +++$+++ " + answer + "\n")
                    j += 1
                question = ""
                answer = ""
                isServe = -1
                chat_id = lines[0]
                session+=1

        if int(lines[2]) == 1 and isServe == 0:
            answer = ""
        if int(lines[2]) == 0 and isServe == 1 and question!="":
            # print(j)
            output_file.write(str(j * 2) + " +++$+++ "+str(session)+ " +++$+++ " + question+"\n")
            output_file.write(str(j * 2 + 1) + " +++$+++ " +str(session)+" +++$+++ " + answer+"\n")
            j+=1
            question = ""
            answer = ""
        if  int(lines[2]) == 1:
            answer += lines[-1]

        if  int(lines[2]) == 0:
            question += lines[-1]
        # print(lines)
        isServe = int(lines[2])

    input_file.close()
    output_file.close()

def transfer(input_folder, output_file):
    """
    tranfer xiaohuangji data
    """
    question_file = open(input_folder+"/question")
    answer_file = open(input_folder+"/answer")
    question, answer = [], []
    for line in question_file.readlines():
        question.append(line)
    for line in answer_file.readlines():
        answer.append(line)
    question_file.close()
    answer_file.close()

    output = open(output_file, 'w')
    for i in range(min(len(question), len(answer))):
        output.write(str(i * 2) + " +++$+++ x +++$+++ " + question[i])
        output.write(str(i * 2 + 1) + " +++$+++ x +++$+++ " + answer[i])
    output.close()



def main():
    preprocess("data/chat.txt", "data/jd_chat.txt")
    # transfer("xiaohuangji", "data/xiaohuangji.txt")


if __name__ == "__main__":
    main()
