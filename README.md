# Reinforcement Learning For Dialogue

### Reinforcement Learning
> - python 2.7
> - tensorflow 1.8.0

### Files
> - QA ------- Submodel
> - Chit-chat ------------ Submodel
> - ReinforcementLearning.py ---------- Main file
> - Environment.py ----------- Environment in RL
> - ModelParametersCopier.py ---- Copy models
> - StateProcessor.py ----------- Generate state embedding
> - Estimator.py ---------------- Q policy neural networks
> - ActionInference.py ---------- Action inference
> - bleu.py --------------------- BLEU score calculate


### Data preprcessing
> - Use data in data/jd_chat.txt, and already split with each session
> - Apply BAIDUBAIKE word vector (in data/single_word_embedding) as initial word embedding
> - If use JD word vector, may perform better, use inference in QA/Data.py read_single_word_embedding() function

### Model RL
details in https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_q_learning.html
> - Question (embedding vector) feeds into Reinforcement learning model 
> - Reinforcement learning model, based on question and Q policy, choose the action [0, 1], 0 means QA, 1 means Chit-chat
> - For the action, generate reply based on the question
> - Compare the reply and real reply, calculate similarity (embedding cosine similarity) as the reward ( +1 * similarity)
> - Environment receive the reply and reward, feedback to the RL model
> - RL model update Q policy. For Q policy, I used one hidden layer with 512 neurons and loss function is (reward + max_a Q' - Q )^2, Q' is based on the next state and the action, Q is based on the  current state and the action
> - Iterate step 1-6

*plus: Data not uploaded beacuse of the privacy*