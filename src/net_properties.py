class NetProperties:
    def __init__(self, word_embed_dim, pos_embed_dim, label_embed_dim, hidden1_dim, hidden2_dim, minibatch_size):
        self.word_embed_dim = word_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.label_embed_dim = label_embed_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.minibatch_size = minibatch_size
