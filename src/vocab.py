from collections import defaultdict

class Vocab:
    def __init__(self, data_paths):
        vocab_word = open(data_paths['word'], 'r').read().strip().split('\n')
        vocab_pos = open(data_paths['pos'], 'r').read().strip().split('\n')
        vocab_label = open(data_paths['label'], 'r').read().strip().split('\n')
        vocab_action = open(data_paths['action'], 'r').read().strip().split('\n')

        self.dict_word_id, self.dict_id_word = defaultdict(int), defaultdict(str)
        for line in vocab_word:
            word, id = line.split(' ')
            self.dict_word_id[word] = int(id)
            self.dict_id_word[int(id)] = word

        self.dict_pos_id, self.dict_id_pos = defaultdict(int), defaultdict(str)
        for line in vocab_pos:
            pos, id = line.split(' ')
            self.dict_pos_id[pos] = int(id)
            self.dict_id_pos[int(id)] = pos

        self.dict_label_id, self.dict_id_label = defaultdict(int), defaultdict(str)
        for line in vocab_label:
            label, id = line.split(' ')
            self.dict_label_id[label] = int(id)
            self.dict_id_label[int(id)] = label

        self.dict_action_id, self.dict_id_action = defaultdict(int), defaultdict(str)
        for line in vocab_action:
            action, id = line.split(' ')
            self.dict_action_id[action] = int(id)
            self.dict_id_action[int(id)] = action

    def word2id(self, word):
        return self.dict_word_id[word]

    def id2word(self, id):
        return self.dict_id_word[id]

    def pos2id(self, pos):
        return self.dict_pos_id[pos]

    def id2pos(self, id):
        return self.dict_id_pos[id]

    def label2id(self, label):
        return self.dict_label_id[label]

    def id2label(self, id):
        return self.dict_id_label[id]

    def action2id(self, action):
        return self.dict_action_id[action]

    def id2action(self, id):
        return self.dict_id_action[id]

    def num_word(self):
        return len(self.dict_word_id)

    def num_pos(self):
        return len(self.dict_pos_id)

    def num_label(self):
        return len(self.dict_label_id)

    def num_action(self):
        return len(self.dict_action_id)