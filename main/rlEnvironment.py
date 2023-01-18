from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")

import numpy as np
import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step

from grammar.sparql import SparqlGrammar

"""KGQA Environment"""
class KGQAEnvironment(PyEnvironment):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 all_questions,
                 all_entities_relations,
                 all_actions,
                 all_answers,
                 question_ids,
                 action_nbrs,
                 discount):
        """
        :param observation_spec: observeration specification
        :param action_spec: action specification
        :param all_questions: question encodings
        :param question_ids: list with all question ids
        :param starts_per_question: context entities per startpoint
        :param q_start_indices: indices for qid and context entity number
        :param all_actions: action encodings
        :param action_nbrs: number of action (= number of paths per context entity)
        :param all_answers: gold label answer
        """

        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._batch_size = 1

        self.all_questions = all_questions
        self.all_answers = all_answers
        self.all_entities_relations = all_entities_relations
        self.all_actions = all_actions
        self.question_ids = question_ids
        self.number_of_actions = action_nbrs

        self.grammar = SparqlGrammar()

        self.question_counter = 0
        self.generated_sparql = ""

        self._done = False

        self.discount = discount

        super(KGQAEnvironment, self).__init__()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._batch_size

    def _empty_observation(self):
        return tf.nest.map_structure(lambda x: np.zeros(x.shape, x.dtype), self.observation_spec())

    def _get_observation(self):
        """Returns an observation"""

        """
        encoded_questions = self.all_encoded_questions[self.current_question_id]
        encoded_keywords
        topic_entities = self.all_topic_entities[self.current_question_id]
        encoded_topic_entities = self.all_encoded_topic_entities[topic_entities]
        known_relations = self.all_known_relations[self.current_question_id]
        encoded_known_relations = self.all_encoded_relations[known_relations]
        
        # create mask 
        
        observation = tf.keras.concatenate([encoded_questions, encoded_topic_entities, encoded_known_relations], axis=0)
        observation = tf.expand_dims(observation, 0)
        observation = tf.keras.layers.concatenate([observation, mask], axis=2)
        tf.dtypes.cast(observation, tf.float32)
        
        return observation
        """

        self.curr_question_id = self.question_ids[self.question_counter]

        # get pre-computed embeddings for the question
        encoded_question = self.all_questions[self.curr_question_id]
        # get pre-computed embeddings for the entities and relations
        encoded_entities = self.all_entities_relations[self.curr_question_id]["entities"]
        encoded_relations = self.all_entities_relations[self.curr_question_id]["relations"]

        self.keywords = self.grammar.keywords()

        # get action embeddings
        encoded_actions = self.all_actions[self.curr_startpoint_id]
        action_nbr = self.number_of_actions[self.curr_startpoint_id]

        mask = tf.ones(action_nbr)
        zeros = tf.zeros((1001-action_nbr))
        mask = tf.keras.layers.concatenate([mask, zeros], axis=0)
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, -1) # [1,1001,1]

        # put them together as next observation for the policy network
        observation = tf.keras.layers.concatenate([encoded_question, encoded_actions], axis=0) # [1001, 768]
        observation = tf.expand_dims(observation, 0) # [1, 1001, 768]
        observation = tf.keras.layers.concatenate([observation, mask], axis=2) #[1, 1001, 769]
        tf.dtypes.cast(observation, tf.float32)

        return observation

    def _reset(self):
        obs = self._get_observation()
        # if self._is_final_observation:
        #     print("final obs inside reset")
        #     return time_step.termination(self._empty_observation(), [0.0])
        return time_step.restart(obs, batch_size=self._batch_size)

    # reset the environment to its initial state
    def reset_env(self):
        self.question_counter = 0
        self.generated_sparql = ""
        self.curr_question_id = ""
        self.curr_startpoint_id = ""

    def _apply_action(self, action):
        """Appies ´action´ to the Environment
        and returns the corresponding reward

        Args:
            action: A value conforming action_spec that will be taken as action in the environment.

        Returns:
            a float value that is the reward received by the environment.
        """

        """
        action = WHERE
        self.generated_sparql = "<sos> SELECT DISTINCT ?ans "
        
        check if the process terminates, like <eos>, 
        if yes, execute the sparql against kg, and get the results and calculate the f1 score,
        if not, then check if the generated sparql is grammatically correct
        if yes, return a positive score for grammar, otherwise, return a negative score for grammar 
        total reward might the sum of f1 score + grammar score 
        """

        return [0.0]

    def _step(self, action):

        reward = self._apply_action(action)

        ts = time_step.termination(self._empty_observation(), reward)
        #ts = time_step.transition(self._get_observation(), reward, discount=self.discount)

        # check if the process terminates by checking the action equal to <eos>

        self._done = False
        return ts