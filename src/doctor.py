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
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

from src.utils import (
    decision_tree_consultation,
    dt_reverse_order_consultation,
    plot_decision_tree,
)


# blocks of questions e.g. baseline questions (age, country, season) and symptom questions
# output is collection of list of tuples containing features (q, a_c, a_gt) and labels (dx)
class BaseDoctor(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abc.abstractmethod
    def conduct_consultation(self, patient):
        pass


class DTBaseDoctor(BaseDoctor):
    """
    Trains decision tree classifier to find best logic for the data, which is applied in different ways
    by profiles defined by derived classes
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs["data"]
        self.doc_kwargs = kwargs["doc"]
        self.categorical_mapping = kwargs["categorical_mapping"]
        self.features = [
            x for x in self.data.columns if x.split("_")[0] in ["base", "symptom"]
        ]
        self.targets = [x for x in self.data.columns if x.split("_")[0] == "disease"]
        self.data_dict = {
            "data": self.data[self.features],
            "target": self.data[kwargs["disease_list"]],
            "feature_names": self.features,
            "target_names": kwargs["disease_list"],
        }

        self.clf = tree.DecisionTreeClassifier(max_depth=kwargs["doc"].max_dt_depth)
        self.clf = self.clf.fit(
            self.data_dict["data"].values, self.data_dict["target"].values
        )


class DecisionTreeDoctor(DTBaseDoctor):
    """
    Doctor that follows decision tree logic
    https://www.youtube.com/watch?v=v68zYyaEmEA
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def conduct_consultation(self, patient):
        # metrics = cross_val_score(clf, data_dict['data'], data_dict['target'], scoring='accuracy', cv=10)
        consultation_block = decision_tree_consultation(
            self.clf, patient, self.features, self.categorical_mapping
        )

        base_feature_block = [
            x for x in consultation_block if x[0].split("_")[0] == "base"
        ]
        symptom_block = [
            x for x in consultation_block if x[0].split("_")[0] == "symptom"
        ]

        # assert all(patient[[q for q, a in consultation_block]] == [a for q, a in consultation_block])

        return {
            "consultation": [base_feature_block, symptom_block],
        }


class DTPoisonerDoctor(DTBaseDoctor):
    """
    Doctor that follows decision tree logic to reach correct diagnosis but may corrupt answers in the process
    https://www.youtube.com/watch?v=v68zYyaEmEA
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def conduct_consultation(self, patient):
        patient_disease = np.array(self.targets)[patient[self.targets] == 1][0]
        consultation_block = dt_reverse_order_consultation(
            self.clf,
            patient,
            self.features,
            self.targets,
            patient_disease,
            self.doc_kwargs.min_correct_ans,
            self.categorical_mapping,
        )

        base_feature_block = [
            x for x in consultation_block if x[0].split("_")[0] == "base"
        ]
        symptom_block = [
            x for x in consultation_block if x[0].split("_")[0] == "symptom"
        ]

        return {
            "consultation": [base_feature_block, symptom_block],
        }


class DTGamerDoctor(DTBaseDoctor):
    """
    Doctor that follows decision tree logic but only to reach a premeditated diagnosis
    https://www.youtube.com/watch?v=v68zYyaEmEA
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def conduct_consultation(self, patient):
        patient_disease = np.random.choice(self.targets)
        consultation_block = dt_reverse_order_consultation(
            self.clf,
            patient,
            self.features,
            self.targets,
            patient_disease,
            self.doc_kwargs.min_correct_ans,
            self.categorical_mapping,
        )
        base_feature_block = [
            x for x in consultation_block if x[0].split("_")[0] == "base"
        ]
        symptom_block = [
            x for x in consultation_block if x[0].split("_")[0] == "symptom"
        ]

        return {
            "consultation": [base_feature_block, symptom_block],
        }


# miscellaneous doctor profiles
class RandomDoctor(BaseDoctor):
    """
    Doctor that selects questions randomly and may randomly get wrong answers
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.categorical_mapping = kwargs["categorical_mapping"]

    def conduct_consultation(self, patient):
        question_inds = np.random.uniform(0, 1, len(self.kwargs["symptom_list"]))
        question_inds = np.where(
            question_inds < self.kwargs["doc"].prob_q_asked, True, False
        )
        questions = np.array(self.kwargs["symptom_list"])[question_inds]
        np.random.shuffle(questions)

        # answers = np.array(patient[questions]) == 'True'
        wrong_answer_inds = np.random.uniform(0, 1, questions.shape)
        wrong_answer_inds = np.where(
            wrong_answer_inds < self.kwargs["doc"].prob_incorrect, False, True
        )
        dodgy_answers = np.empty(wrong_answer_inds.shape).astype(np.int)
        for i, answer in enumerate(wrong_answer_inds):
            # if symptom incorrectly diagnosed
            if not answer:
                # if symptom categorical/ continuous
                if questions[i] in self.categorical_mapping:
                    potential_wrong_answers = [
                        x
                        for x in self.categorical_mapping[questions[i]].keys()
                        if not x == patient[questions[i]]
                    ]
                    dodgy_answers[i] = np.random.choice(potential_wrong_answers)
                # otherwise it's binary
                else:
                    dodgy_answers[i] = (
                        1 - patient[questions[i]]
                    )  # incorrect binary symptom
            else:
                dodgy_answers[i] = patient[questions[i]]  # correct symptom

        symptom_str_answers = [
            self.categorical_mapping[q][a]
            if q in self.categorical_mapping
            else ["False", "True"][a]
            for q, a in zip(questions, dodgy_answers)
        ]
        symptom_block = [
            (q, a_rand) for q, a_rand in zip(questions, symptom_str_answers)
        ]

        base_feature_qs = self.kwargs["base_feature_list"]
        base_feature_str_answers = [
            self.categorical_mapping[q][a]
            if q in self.categorical_mapping
            else ["False", "True"][a]
            for q, a in zip(base_feature_qs, patient[base_feature_qs])
        ]
        base_feature_block = [
            (q, a) for q, a in zip(base_feature_qs, base_feature_str_answers)
        ]

        return {
            "consultation": [base_feature_block, symptom_block],
        }


class BiasedDoctor(BaseDoctor):
    """
    Doctor that never asks certain questions (can select which questions are not asked randomly)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs["data"]
        self.categorical_mapping = kwargs["categorical_mapping"]

    def conduct_consultation(self, patient):
        # guarantee the biased questions will be asked, ask the rest with certain probabilities left over
        biased_symptom_qs = np.array(self.kwargs["doc"].biased_symptoms)
        remaining_features = self.kwargs["base_feature_list"] + [
            x for x in self.kwargs["symptom_list"] if not x in biased_symptom_qs
        ]

        question_inds = np.random.uniform(0, 1, len(remaining_features))
        question_inds = np.where(
            question_inds < self.kwargs["doc"].prob_other_q_asked, True, False
        )

        questions = np.concatenate(
            [biased_symptom_qs, np.array(remaining_features)[question_inds]]
        )
        answers = patient[questions]
        str_answers = [
            self.categorical_mapping[q][a]
            if q in self.categorical_mapping
            else ["False", "True"][a]
            for q, a in zip(questions, answers)
        ]

        symptom_block = [
            (q, a) for q, a in zip(questions, str_answers) if "symptom" in q
        ]
        base_feature_block = [
            (q, a) for q, a in zip(questions, str_answers) if "base" in q
        ]

        return {
            "consultation": [base_feature_block, symptom_block],
        }


class NoviceDoctor(BaseDoctor):
    """
    Doctor that sometimes gets wrong answers
    """

    def __init__(self, data):
        self.data = data

    def conduct_consultation(self):
        raise NotImplementedError


class ComprehensiveDoctor(BaseDoctor):
    """
    Doctor that asks all questions in the same order as the dataframe and correctly records all symptoms
    """

    def __init__(self, data):
        super().__init__(**kwargs)

    def conduct_consultation(self):
        raise NotImplementedError
