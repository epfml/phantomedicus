"""Script for preprocessing existing consultation data to generate Bayesian Network"""
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import embed
from munch import munchify

from config import load_config
from src.utils import pickle_object

# TODO: automate parsing based on datatypes e.g. binning for continuous vals
# TODO: test out parameter learning functionality of pgmpy to learn CPDs based on a predefined probabilistic DAG
# -> https://pgmpy.org/examples/Learning%20Parameters%20in%20Discrete%20Bayesian%20Networks.html?highlight=parameter%20learning


def extract_column_data_type(dict_csv):
    pass


def update_dict(prob_dict, source, target, probabilities):
    if source in prob_dict.keys():
        if prob_dict[source]:
            prob_dict[source][target] = np.array(probabilities)
        else:
            prob_dict[source] = {}
            prob_dict[source][target] = np.array(probabilities)
    else:
        prob_dict[source] = {}
        prob_dict[source][target] = np.array(probabilities)

    return prob_dict


def preprocess_df(df, prev_base_features, new_base_features, prev_diseases, new_diseases, prev_symptoms, new_symptoms):
    prev_names = prev_base_features + prev_diseases + prev_symptoms
    new_names = new_base_features + new_diseases + new_symptoms
    df.rename(columns={prev: new for prev, new in zip(prev_names, new_names)}, inplace=True)

    replace_dict = {"none": np.nan, "normal": np.nan}
    df["symptom_temp"].replace(",", ".", regex=True, inplace=True)
    df["base_a_weight"].replace(",", ".", regex=True, inplace=True)
    df["base_a_age"].replace(",", ".", regex=True, inplace=True)
    df["base_wfa"].replace(",", ".", regex=True, inplace=True)
    df["base_wfh2"].replace(",", ".", regex=True, inplace=True)
    df.replace(replace_dict, inplace=True)
    non_numeric_columns = ["base_country", "base_a_gender2"]
    numeric_columns = [x for x in df.columns if not x in non_numeric_columns]
    df[numeric_columns] = df[numeric_columns].astype(np.float16)

    return df


if __name__=="__main__":
    with open("metadata.json") as f:
        metadata = json.load(f)
    metadata = munchify(metadata)
    cfg = load_config(metadata)

    _base_features = [
        "country", "a_age", "a_gender2", "a_weight", "wfa", "wfh2"
    ]
    base_features = ["base_" + x for x in _base_features]

    _considered_diseases = ['classify_cough_pneumonia', 'classify_ear_acute_infection', 'classify_abdo_gastro',
                            'classify_skin_scabies_simple', 'classify_skin_tinea_capitis', 'classify_malnut_mam',
                            'classify_abdo_unclassified', 'classify_malaria_severe','classify_skin_eczema_simple',
                            'classify_mouth_oral_trush']

    considered_diseases = ["disease_" + "_".join(x.split("_")[1:]) for x in _considered_diseases]

    _considered_symptoms = [
        "s_diarr", "s_oedema", "s_drepano", "s_fever_temp", "s_fever_his", "s_temp",
        "s_cough",  "s_abdopain", "s_vomit", "s_skin", "s_earpain", "s_eyepb", "s_throat",
        "s_mouthpb", "s_dysuria", "s_hematuria", "s_joint", "s_limp", "ms_measles"
    ]
    considered_symptoms = ["symptom_" + "_".join(x.split("_")[1:]) for x in _considered_symptoms]

    df = pd.read_csv(cfg.consultation_data_path)
    df = df[_base_features + _considered_diseases + _considered_symptoms]
    df = preprocess_df(df, _base_features, base_features, _considered_diseases, considered_diseases, _considered_symptoms, considered_symptoms)
    embed()
    base_features_state_dict = {
        "base_country":
        {
            "dtype": "categorical",
            "state_names": ["rca", "Mali", "Kenya", "Tanzania", "Niger", "Nigeria", "Tchad"],
            "vals": ["rca", "Mali", "Kenya", "Tanzania", "Niger", "Nigeria", "Tchad"],
            "prob": [1 / 7] * 7
        },
        "base_a_age":
        {
            "dtype": "continuous",
            "state_names": ["0 - 12", "12 - 24", "24 - 48", "48 - 61"],
            "vals": [0, 12, 24, 48, 61],
            "prob": [1 / 4] * 4
        },
        "base_a_gender2":
        {
            "dtype": "categorical",
            "state_names": ["male", "female"],
            "vals": ["male", "female"],
            "prob": [1 / 2] * 2
        },
        "base_a_weight":
        {
            "dtype": "continuous",
            "state_names": ["2 - 5", "5 - 10", "10 - 15", "15 - 20", "20 - 30"],
            "vals": [2, 5, 10, 15, 20, 30],
            "prob": [1 / 5] * 5
        },
        "base_wfa":
        {
            "dtype": "continuous",
            "state_names": ["-3 - -0.5", "-0.5 - 0.5", "0.5 - 4"],
            "vals": [-3, -0.5, 0.5, 3],
            "prob": [1 / 3] * 3
        },
        "base_wfh2":
        {
            "dtype": "continuous",
            "state_names": ["-3 - -0.5", "-0.5 - 0.5", "0.5 - 4"],
            "vals": [-3, -0.5, 0.5, 4],
            "prob": [1 / 3] * 3
        }
    }
    base_features_state_dict = munchify(base_features_state_dict)

    disease_state_dict = {k: {"state_names": ["False", "True"], "dtype": "binary"} for k in considered_diseases}
    symptom_state_dict = {k: {"state_names": ["False", "True"], "dtype": "binary"} for k in considered_symptoms if not k == "symptom_temp"}
    fever_bins = [35, 37, 40, 44]
    fever_state_names = [str(x) + "-" + str(y) for x, y in zip(fever_bins[:-1], fever_bins[1:])]
    symptom_state_dict["symptom_temp"] = {"state_names": fever_state_names, "vals": fever_bins, "dtype": "continuous"}

    base_feature_disease_prob_dict = {}
    for disease in tqdm(considered_diseases):
        for base_feature in base_features:
            base_feature_dict = base_features_state_dict[base_feature]
            states = base_feature_dict.vals
            if base_feature_dict.dtype == "continuous":
                prob_disease_given_base_bins = []
                pos_disease_inds = df[disease] == 1
                for i in range(len(states) - 1):
                    state_interval_inds = (df[base_feature] > states[i]) & (df[base_feature] <= states[i + 1])
                    prob_disease_given_base_bin = sum(state_interval_inds & pos_disease_inds) / sum(state_interval_inds)
                    prob_disease_given_base_bins.append(prob_disease_given_base_bin)

                if not all([v == 0 for v in prob_disease_given_base_bins]):
                    base_feature_disease_prob_dict = update_dict(base_feature_disease_prob_dict, base_feature, disease, prob_disease_given_base_bins)

            elif base_feature_dict.dtype in ["binary", "categorical"]:
                prob_disease_given_states = []
                # prob_disease_given_not_state = []

                pos_disease_inds = df[disease] == 1
                for state in states:
                    state_inds = df[base_feature] == state

                    # proportion of patients with base feature that also have disease - we still take NaNs into account
                    prob_disease_given_state = sum(df[state_inds][disease] == 1) / sum(state_inds)
                    prob_no_disease_given_state = 1 - prob_disease_given_state
                    assert (prob_disease_given_state <= 1) & (prob_no_disease_given_state <= 1), f"prob of having/ not having {disease} greater than 1"
                    prob_disease_given_states.append(prob_disease_given_state)

                if prob_disease_given_state > 0:
                    base_feature_disease_prob_dict = update_dict(base_feature_disease_prob_dict, base_feature, disease, prob_disease_given_states)

    # filter symptoms/diseases that have less than a certain amount of valid entries
    df_diseases = df[considered_diseases]
    df_symptoms = df[considered_symptoms]

    # here would need to figure out way to automatically infer RV type and extract probabilities accordingly
    disease_symptom_prob_dict = {}
    for symptom in tqdm(considered_symptoms):
        symptom_dict = symptom_state_dict[symptom]
        for disease in considered_diseases:
            if symptom_dict['dtype'] == "binary":
                # extract indices corresponding to positive cases of disease, determine probability of disease based on that
                vals, inds = np.unique(df[disease], return_inverse=True)
                pos_inds = inds == 1
                assert np.all(np.array(df[pos_inds][disease]) == 1), "not all negative indices correspond to negative cases"

                # proportion of patients with disease that also have symptom
                prob_symptom_with_disease = sum(df[pos_inds][symptom] == 1) / sum(pos_inds)
                assert (prob_symptom_with_disease <= 1), f"prob of having {symptom} greater than 1"

                if prob_symptom_with_disease > 0:
                    disease_symptom_prob_dict = update_dict(disease_symptom_prob_dict, disease, symptom, prob_symptom_with_disease)

            elif symptom_dict['dtype'] == "continuous":
                num_valid = len(df[disease].dropna())
                num_pos = sum(df[disease] == 1)
                prob_symptom_within_fever_range = []
                pos_disease_inds = df[disease] == 1
                for i in range(len(fever_bins) - 1):
                    fever_inds = (df[symptom] > fever_bins[i]) & (df[symptom] <= fever_bins[i + 1])
                    p = sum(df[fever_inds & pos_disease_inds][disease] == 1) / num_pos
                    prob_symptom_within_fever_range.append(p)

                if not all([v == 0 for v in prob_symptom_within_fever_range]):
                    disease_symptom_prob_dict = update_dict(disease_symptom_prob_dict, disease, symptom, prob_symptom_within_fever_range)

    doctors = {
        "types": ["decision_tree", "random", "biased"],
        "country":
        {
            "rca": {
                "prob_doctor": [0.6, 0.2, 0.2],
                "doctor_kwargs":
                {
                    "decision_tree": {
                        "max_dt_depth": 10
                    },
                    "random": {
                        "prob_q_asked": 0.8,
                        "prob_incorrect": 0.05
                    },
                    "biased":
                    {
                        "prob_other_q_asked": 0.5,
                        "biased_symptoms": ["symptom_cough", "symptom_fever_his", "symptom_diarr", "symptom_abdopain"]
                    }
                }
            },
            "Mali": {
                "prob_doctor": [0.6, 0.2, 0.2],
                "doctor_kwargs":
                {
                    "decision_tree": {
                        "max_dt_depth": 10
                    },
                    "random": {
                        "prob_q_asked": 0.8,
                        "prob_incorrect": 0.05
                    },
                    "biased":
                    {
                        "prob_other_q_asked": 0.5,
                        "biased_symptoms": ["symptom_cough", "symptom_fever_his", "symptom_diarr", "symptom_skin"]
                    }
                }
            },
            "Kenya": {
                "prob_doctor": [0.6, 0.2, 0.2],
                "doctor_kwargs":
                {
                    "decision_tree": {
                        "max_dt_depth": 10
                    },
                    "random": {
                        "prob_q_asked": 0.8,
                        "prob_incorrect": 0.05
                    },
                    "biased":
                    {
                        "prob_other_q_asked": 0.5,
                        "biased_symptoms": ["symptom_cough", "symptom_fever_his", "symptom_diarr", "symptom_skin"]
                    }
                }
            },
            "Tanzania": {
                "prob_doctor": [0.6, 0.2, 0.2],
                "doctor_kwargs":
                {
                    "decision_tree": {
                        "max_dt_depth": 10
                    },
                    "random": {
                        "prob_q_asked": 0.8,
                        "prob_incorrect": 0.05
                    },
                    "biased":
                    {
                        "prob_other_q_asked": 0.5,
                        "biased_symptoms": ["symptom_cough", "symptom_fever_his", "symptom_diarr", "symptom_fever_temp"]
                    }
                }
            },
            "Niger": {
                "prob_doctor": [0.6, 0.2, 0.2],
                "doctor_kwargs":
                {
                    "decision_tree": {
                        "max_dt_depth": 10
                    },
                    "random": {
                        "prob_q_asked": 0.8,
                        "prob_incorrect": 0.05
                    },
                    "biased":
                    {
                        "prob_other_q_asked": 0.5,
                        "biased_symptoms": ["symptom_cough", "symptom_fever_his", "symptom_diarr", "symptom_skin"]
                    }
                }
            },
            "Nigeria": {
                "prob_doctor": [0.6, 0.2, 0.2],
                "doctor_kwargs":
                {
                    "decision_tree": {
                        "max_dt_depth": 10
                    },
                    "random": {
                        "prob_q_asked": 0.8,
                        "prob_incorrect": 0.05
                    },
                    "biased":
                    {
                        "prob_other_q_asked": 0.5,
                        "biased_symptoms": ["symptom_cough", "symptom_fever_his", "symptom_diarr", "symptom_skin"]
                    }
                }
            },
            "Tchad": {
                "prob_doctor": [0.6, 0.2, 0.2],
                "doctor_kwargs":
                {
                    "decision_tree": {
                        "max_dt_depth": 10
                    },
                    "random": {
                        "prob_q_asked": 0.8,
                        "prob_incorrect": 0.05
                    },
                    "biased":
                    {
                        "prob_other_q_asked": 0.5,
                        "biased_symptoms": ["symptom_cough", "symptom_fever_his", "symptom_diarr", "symptom_skin"]
                    }
                }
            }
        }
    }

    metadata_dict = {
        "disease_list": considered_diseases,
        "symptom_list": considered_symptoms,
        "node_states": {
            "patient_attributes": base_features_state_dict,
            "diseases": disease_state_dict,
            "symptoms": symptom_state_dict
        },
        "patient_attribute_disease_probs": base_feature_disease_prob_dict,
        "disease_symptom_probs": disease_symptom_prob_dict,
        "doctors": doctors
    }

    pickle_object(metadata_dict, cfg.consultation_data_prob_dict_path)

    embed()
