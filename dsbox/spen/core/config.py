

class Config:
  def __init__(self):
    self.dropout = 0.0
    self.layer_info = None
    self.en_layer_info = None
    self.weight_decay = 0.0001
    self.l2_penalty = 0.0
    self.en_variable_scope = "en"
    self.fx_variable_scope = "fx"
    self.spen_variable_scope = "spen"
    self.inf_rate = 0.1

    self.learning_rate = 0.001
    self.inf_iter = 10
    self.dropout = 0.0
    self.train_iter = 10
    self.batch_size = 100
    self.dimension = 2
    self.filter_sizes = [2,3,4,5]
    self.num_filters = 10
    self.num_samples = 10
    self.margin_weight = 100.0
    self.exploration = 0.0
    self.lstm_hidden_size = 100
    self.vocabulary_size = 20608
    self.embedding_size = 100
    self.sequence_length = 118
    self.hidden_num = 100
    self.input_num = 0
    self.output_num = 0
