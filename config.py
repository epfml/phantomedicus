"""Configuration parameters"""
import numpy as np

from src.doctor import *


class Config:
    def __init__(self):
        super().__init__()


def load_config():
    cfg = Config()

    # set to True if you want to extract statistics from existing consultation data, and provide the path
    cfg.generate_from_consultation_data = True
    cfg.consultation_data_path = "../data/all_preprocessed.csv"
    cfg.consultation_data_prob_dict_path = "data/consultation_data_prob_dict.pkl"
    cfg.consultation_output_dir = "data/consultations/"

    cfg.n_clinics = 10
    cfg.n_patients_per_clinic = 600

    cfg.doctor_dict = {
        "decision_tree": DecisionTreeDoctor,
        "decision_tree_poisoner": DTPoisonerDoctor,
        "decision_tree_gamer": DTGamerDoctor,
        "random": RandomDoctor,
        "biased": BiasedDoctor,
    }

    return cfg
