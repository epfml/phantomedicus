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

from utils import decision_tree_consultation, plot_decision_tree


# blocks of questions e.g. baseline questions (age, country, season) and symptom questions
# output is collection of list of tuples containing features (q, a_c, a_gt) and labels (dx)
class BaseDoctor(metaclass=abc.ABCMeta):
    def __init__(self, data):
        self.data = data
        self.symptoms = np.array([x for x in self.data.columns if 'symptom' in x])
        self.diseases = np.array([x for x in self.data.columns if 'disease' in x])
        self.consultation = []

    def save_consultation(self, path):
        assert len(self.consultation) != 0, "You need to run 'conduct_consultation' before saving..."

        if not (path.split(".")[-1] == "pickle"):
            path += ".pickle"

        with open(f"{path}", "wb") as f:
            pickle.dump(self.consultation, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Consultation saved to {path}")

    @abc.abstractmethod
    def conduct_consultation(self):
        pass


class PerfectDoctor(BaseDoctor):
    """
    Doctor that asks all questions in the same order as the dataframe and correctly records all symptoms
    """
    def __init__(self, data):
        super().__init__()

    def conduct_consultation(self):
        raise NotImplementedError


class RandomDoctor(BaseDoctor):
    """
    Doctor that selects questions randomly and may randomly get wrong answers
    """
    def __init__(self, data):
        super().__init__(data)

    def conduct_consultation(self):
        for ind, row in self.data.iterrows():
            question_inds = np.random.randint(0, 2, self.symptoms.shape) == 1
            questions = self.symptoms[question_inds]
            np.random.shuffle(questions)

            answers = np.array(row[questions], dtype=bool)
            wrong_answer_inds = np.random.randint(0, 2, questions.shape) == 1
            wrong_answers = np.where(wrong_answer_inds == 1, ~answers, answers)

            question_answer_pairs = [(q, a_rand, a_true) for q, a_rand, a_true in zip(questions, wrong_answers, answers)]
            diseases = row[self.diseases].to_dict()

            self.consultation.append((question_answer_pairs, diseases))

        return self.consultation


class DecisionTreeDoctor(BaseDoctor):
    """
    Doctor that follows decision tree logic
    https://www.youtube.com/watch?v=v68zYyaEmEA
    """
    def __init__(self, data):
        super().__init__(data)
        self.data = data

    def conduct_consultation(self):
        # first we train decision tree classifier to determine best logic for the dataset, and apply this repeatedly to each patient
        self.data_dict = {'data': self.data[self.symptoms].values,
                          'target': self.data[self.diseases].values,
                          'feature_names': self.symptoms,
                          'target_names': self.diseases}

        X, y = self.data_dict['data'], self.data_dict['target']
        self.clf = tree.DecisionTreeClassifier()
        self.clf = self.clf.fit(X, y)
        # metrics = cross_val_score(clf, data_dict['data'], data_dict['target'], scoring='accuracy', cv=10)

        for ind, patient in self.data.iterrows():
            self.consultation.append(decision_tree_consultation(self.clf, patient[self.symptoms], self.symptoms))

        return self.consultation


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
    embed()
    dt_doctor = DecisionTreeDoctor(data)
    dt_consultation = dt_doctor.conduct_consultation()
    dt_doctor.save_consultation("data/dt_consultation.pickle")

    embed()

