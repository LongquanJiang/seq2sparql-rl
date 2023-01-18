
import pickle
import tensorflow as tf
from tf_agents.networks.network import DistributionNetwork
from tf_agents.utils import nest_utils
from tf_agents.networks.utils import BatchSquash
from tf_agents.distributions import masked
import tensorflow_probability as tfp
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# Define the State-Action Policy Network
class PolicyNetwork(DistributionNetwork):

    def __init__(self,
                 seed_value,
                 input_tensor_spec,
                 output_tensor_spec,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='PolicyNetwork'):

        self._out_tensor_spec = output_tensor_spec
        
        super(PolicyNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            output_spec=tfp.distributions.Categorical,
            name=name
        )

        self.num_keywords = 50
        self.num_variables = 5
        self.num_entities = 5
        self.num_relations = 5
        self.keywords_emb = tf.keras.layers.Embedding(input_dim=self.num_keywords, output_dim=768, trainable=True)
        self.variables_emb = tf.keras.layers.Embedding(input_dim=self.num_variables, output_dim=768, trainable=True)
        self.kg_model = pickle.load(open("transe_wikidata5m.pkl", "rb"))
        self.entity2id = self.kg_model.graph.entity2id
        self.relation2id = self.kg_model.graph.relation2id
        self.entity_embeddings = self.kg_model.solver.entity_embeddings
        self.relation_embeddings = self.kg_model.solver.relation_embeddings
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_encoding = TFBertModel.from_pretrained('bert-base-uncased', return_dict=True)

        # entity2id = self.kg_model.graph.entity2id
        # relation2id = self.kg_model.graph.relation2id
        # entity_embeddings = self.kg_model.solver.entity_embeddings
        # relation_embeddings = self.kg_model.solver.relation_embeddings
        # import graphvite as gv
        # alias2entity = gv.dataset.wikidata5m.alias2entity
        # alias2relation = gv.dataset.wikidata5m.alias2relation
        # print(entity_embeddings[entity2id[alias2entity["machine learning"]]])
        # print(relation_embeddings[relation2id[alias2relation["field of work"]]])


        # get same initial values for same seed to make results reproducible
        initializer1 = tf.keras.initializers.GlorotUniform(seed=seed_value)
        initializer2 = tf.keras.initializers.GlorotUniform(seed=(seed_value + 1))

        # define network
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(768, activation=tf.nn.relu, name="dense1", kernel_initializer=initializer1)
        self.dense2 = tf.keras.layers.Dense(768, name="dense2", kernel_initializer=initializer2)

        self.dist = 0
        self.logits = 0


    def encode_question(self, question):
        tokenized_input = self.bert_tokenizer(question, return_tensors="tf", padding=True)
        encoded_input = self.bert_encoding(tokenized_input, output_hidden_states=True)
        # take average over all hidden layers
        all_layers = [encoded_input.hidden_states[l] for l in range(1, 13)]
        encoder_layer = tf.concat(all_layers, 1)
        pooled_output = tf.reduce_mean(encoder_layer, axis=1)

        return pooled_output  # [1, 768]

    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec

    def get_distribution(self):
        return self.dist

    def get_logits(self):
        return self.logits

    def call(self, observations, step_type, network_state, training=False, mask=None):

        """
        get prediction from policy network
        this is called for collecting experience to get the distribution the agent can sample from and
        called once again to get the distribution for a given time step when calculating the loss

        observations=[keywords, variables, relations, entities]

        """

        is_empty = tf.equal(tf.size(observations), 0)
        if is_empty:
            return 0, network_state

        question, keywords, variables, relations, entities = observations

        encoded_question = self.encode_question(question)   # [1, 768]
        encoded_keywords = self.keywords_emb(keywords)      # [num_keywords, embed_size]
        encoded_variables = self.variables_emb(variables)   # [num_variables, embed_size]

        encoded_entities = self.entity_embeddings[[self.entity2id[entity] for entity in entities]]  # [len(entities, 512)]
        encoded_relations = self.relationship_embeddings[[self.relation2id[relation] for relation in relations]]    # [len(relations), 512]

        # Note: dimensionality inconsistent

        observations = tf.keras.layers.concatenate([encoded_question,
                                                    encoded_entities,
                                                    encoded_relations],
                                                   axis=0) # [1+len(entities)+len(relations), 768/512]

        # outer rank will be 0 for one observation, if we have several for calculating the loss it is greater than 1
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        observations = tf.nest.map_structure(BatchSquash(outer_rank).flatten, observations)  # [batch_size, 1+len(entities)+len(relations), 768/512]

        actions = tf.keras.layers.concatenate([encoded_keywords,
                                               encoded_variables,
                                               encoded_entities,
                                               encoded_relations],
                                              axis=0)  # [num_keywords+num_variables+len(entities)+len(relations), 768/512]

        # outer rank will be 0 for one observation, if we have several for calculating the loss it is greater than 1
        outer_rank = nest_utils.get_outer_rank(actions, self._out_tensor_spec)
        actions = tf.nest.map_structure(BatchSquash(outer_rank).flatten, actions)  # [batch_size, num_keywords+num_variables+len(entities)+len(relations), 768/512]

        entity_mask = tf.keras.layers.concatenate([tf.ones(len(entities)), tf.zeros(self.num_entities-len(entities))], axis=0)
        entity_mask = tf.expand_dims(entity_mask, 0)
        entity_mask = tf.expand_dims(entity_mask, -1) # [1, num_entities, 1]

        relation_mask = tf.keras.layers.concatenate([tf.ones(len(relations)), tf.zeros(self.num_relations-len(relations))], axis=0)
        relation_mask = tf.expand_dims(relation_mask, 0)
        relation_mask = tf.expand_dims(relation_mask, -1) # [1, num_relations, 1]

        mask = tf.keras.layers.concatenate([tf.ones([self.num_keywords+self.num_variables]), entity_mask, relation_mask], axis=1) # [1, num_keywords+num_variables+num_entities+num_relations, 1]

        observations = self.flatten(observations)
        x = self.dense1(observations)
        out = self.dense2(x) # [1, 768]
        out = tf.expand_dims(out, -1)

        availableActions = tf.transpose(actions, perm=[0, 2, 1])  # [batchsize, 768, 1000]
        # we multiply actions and output of network and get a matrix where each column is vector for one action, we sum over each column to get score for each action
        scores = tf.reduce_sum(tf.multiply(availableActions, out), 1) # [batchsize, 1000, 1]
        self.logits = scores

        # prepare the mask
        mask = tf.squeeze(mask)
        mask_zero = tf.zeros_like(mask) # (score>0 and score<0)
        mask = tf.math.not_equal(mask, mask_zero)
        mask = tf.transpose(mask)

        mask = mask[:-1]

        mask = tf.transpose(mask)

        # we convert it to categorical distribution, an action will be sampled from it
        # we use a masking distribution here because we can have less than 1000 valid actions, invalid ones are masked out
        self.dist = masked.MaskedCategorical(logits=scores, mask=mask)
        return self.dist, network_state