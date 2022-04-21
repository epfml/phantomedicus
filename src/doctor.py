"""Simulates doctor consultation based on ground truth patient data (comprising base features, symptoms, and diseases)"""
import abc
import pickle

import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from IPython import embed
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from src.utils import decision_tree_consultation, plot_decision_tree


# blocks of questions e.g. baseline questions (age, country, season) and symptom questions
# output is collection of list of tuples containing features (q, a_c, a_gt) and labels (dx)
class BaseDoctor(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.kwargs = kwargs


    @abc.abstractmethod
    def conduct_consultation(self, patient):
        pass


class CuriousDoctor(BaseDoctor):
    """
    Doctor that asks all questions in the same order as the dataframe and correctly records all symptoms
    """
    def __init__(self, data):
        super().__init__(**kwargs)

    def conduct_consultation(self):
        raise NotImplementedError


class RandomDoctor(BaseDoctor):
    """
    Doctor that selects questions randomly and may randomly get wrong answers
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def conduct_consultation(self, patient):
        question_inds = np.random.uniform(0, 1, self.kwargs['symptoms'].shape)
        question_inds = np.where(question_inds < self.kwargs['rand_doc_carelessness'], False, True)
        questions = self.kwargs['symptoms'][question_inds]
        np.random.shuffle(questions)

        answers = np.array(patient[questions]) == 'True'
        wrong_answer_inds = np.random.uniform(0, 1, questions.shape)
        wrong_answer_inds = np.where(wrong_answer_inds < self.kwargs['rand_doc_incorrectness'], 1, 0)
        dodgy_answers = np.where(wrong_answer_inds == 1, ~answers, answers)

        symptom_block = [(q, a_rand) for q, a_rand in zip(questions, dodgy_answers)]
        base_feature_block = [(q, a) for q, a in zip(self.kwargs['base_features'], patient[self.kwargs['base_features']])]

        return {"consultation": [base_feature_block, symptom_block],
                "raw_patient_data": patient}


class DecisionTreeDoctor(BaseDoctor):
    """
    Doctor that follows decision tree logic
    https://www.youtube.com/watch?v=v68zYyaEmEA
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs['data']
        self.data_dict = {'data': np.where(self.data[kwargs['symptoms']].values == 'True', 1, 0),
                          'target': np.where(self.data[kwargs['diseases']].values == 'True', 1, 0),
                          'feature_names': kwargs['symptoms'],
                          'target_names': kwargs['diseases']}

        self.clf = tree.DecisionTreeClassifier(max_depth=kwargs['max_dt_depth'])
        self.clf = self.clf.fit(self.data_dict['data'], self.data_dict['target'])


    def conduct_consultation(self, patient):
        # train decision tree classifier to find best logic for the data, and apply this repeatedly to each patient
        # metrics = cross_val_score(clf, data_dict['data'], data_dict['target'], scoring='accuracy', cv=10)
        base_feature_block = [(q, a) for q, a in zip(self.kwargs['base_features'], patient[self.kwargs['base_features']])]
        symptom_block = decision_tree_consultation(self.clf, patient[self.data_dict['feature_names']], self.data_dict['feature_names'])

        return {"consultation": [base_feature_block, symptom_block],
                "raw_patient_data": patient}


class BiasedDoctor(BaseDoctor):
    """
    Doctor that never asks certain questions (can select which questions are not asked randomly)
    """
    def __init__(self, data):
        self.data = data

    def conduct_consultation(self):
        raise NotImplementedError


class IneptDoctor(BaseDoctor):
    """
    Doctor that sometimes gets wrong answers
    """
    def __init__(self, data):
        self.data = data

    def conduct_consultation(self):
        raise NotImplementedError


if __name__ == "__main__":
    data = pd.read_csv("data/patient_data.csv")
   # random_doctor = RandomDoctor(data)
   # rnd_consultation = random_doctor.conduct_consultation()
   # random_doctor.save_consultation("data/random_consultation.pickle")

    dt_doctor = DecisionTreeDoctor(data)
    dt_consultation = dt_doctor.conduct_consultation()
    dt_doctor.save_consultation("data/dt_consultation.pickle")

    embed()

