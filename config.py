"""Configuration parameters"""
import numpy as np

from src.doctor import *


class Config():
    def __init__(self):
        super().__init__()


def load_config(metadata):
    cfg = Config()


    cfg.n_clinics = 50
    cfg.n_patients_per_clinic = 1000

    cfg.doctor_dict = \
        {
            "decision_tree": DecisionTreeDoctor,
            "random": RandomDoctor
        }

    cfg.doctor_kwargs = \
        {
            'symptoms': np.array(metadata.symptom_list),
            'diseases': np.array(metadata.disease_list),
            'base_features': np.array(list(metadata.node_states.patient_attributes.keys())),
            'max_dt_depth': 6,
            'rand_doc_carelessness': 0.3,
            'rand_doc_incorrectness': 0.2
        }



    return cfg

