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


def parameter_learning(PDAG, filtered_df):
    pass


def update_dict(prob_dict, source, target, probabilities):
    if target in prob_dict.keys():
        if prob_dict[target]:
            prob_dict[target][source] = np.array(probabilities)
        else:
            prob_dict[target] = {}
            prob_dict[target][source] = np.array(probabilities)
    else:
        prob_dict[target] = {}
        prob_dict[target][source] = np.array(probabilities)

    return prob_dict


def preprocess_df(df, prev_base_features, new_base_features, prev_diseases, new_diseases, prev_symptoms, new_symptoms):
    prev_names = prev_base_features + prev_diseases + prev_symptoms
    new_names = new_base_features + new_diseases + new_symptoms
    df.rename(columns={prev: new for prev, new in zip(prev_names, new_names)}, inplace=True)

    replace_dict = {'none': np.nan, 'normal': np.nan}
    df['symptom_temp'].replace(',', '.', regex=True, inplace=True)
    df['base_a_weight'].replace(',', '.', regex=True, inplace=True)
    df['base_a_age'].replace(',', '.', regex=True, inplace=True)
    df['base_wfa'].replace(',', '.', regex=True, inplace=True)
    df['base_wfh2'].replace(',', '.', regex=True, inplace=True)
    df.replace(replace_dict, inplace=True)
    non_numeric_columns = ['base_country', 'base_a_gender2']
    numeric_columns = [x for x in df.columns if not x in non_numeric_columns]
    df[numeric_columns] = df[numeric_columns].astype(np.float16)

    return df


if __name__=="__main__":
    with open("metadata.json") as f:
        metadata = json.load(f)
    metadata = munchify(metadata)
    cfg = load_config(metadata)

    _base_features = [
        'country', 'a_age', 'a_gender2', 'a_weight', 'wfa', 'wfh2'
    ]
    base_features = ['base_' + x for x in _base_features]

    _considered_diseases = [
        'classify_diarrhoea_persistent', 'classify_uti_febrile', 'classify_abdo_constipation', 'classify_abdo_gastro',
        'classify_abdo_resp_pain', 'classify_abdo_unclassified', 'classify_anaphylaxis',
        'classify_anaphylaxis_severe', 'classify_anemia_non_severe', 'classify_anemia_severe',
        'classify_cough_bronchiolitis', 'classify_cough_persistent', 'classify_cough_pneumonia',
        'classify_cough_urti', 'classify_cystitis_acute', 'classify_dehydration_moderate',
        'classify_dehydration_none',  'classify_dysentery_non_compl',
        'classify_ear_acute_infection', 'classify_ear_chronic_discharge','classify_ear_mastoiditis',
        'classify_ear_unclassified', 'classify_eye_unclassified', 'classify_fever_persistent',
        'classify_likely_viral_oma', 'classify_malaria_severe', 'classify_malaria_simple_new', 'classify_malnut_2ds_only__6m',
        'classify_measles_non_compl', 'classify_malnut_mam', 'classify_mouth_lesion_benign', 'classify_mouth_oral_trush',
        'classify_mouth_stomatitis', 'classify_pharyngitis_viral_', 'classify_possible_trachoma', 'classify_severe_abdo_pain',
        'classify_severe_resp_illness', 'classify_skin_abscess_multiple', 'classify_skin_abscess_single',
        'classify_skin_eczema_infected', 'classify_skin_eczema_simple', 'classify_skin_fungal', 'classify_skin_furoncle_simple',
        'classify_skin_herpes', 'classify_skin_molluscum', 'classify_skin_scabies_infected', 'classify_skin_scabies_simple',
        'classify_skin_tinea_capitis', 'classify_skin_urticaria', 'classify_skin_zona', 'classify_sam_with_complications',
        'classify_uti_upper'
    ]
    considered_diseases = ["disease_" + "_".join(x.split("_")[1:]) for x in _considered_diseases]

    _considered_symptoms = [
        's_diarr', 's_oedema','s_temp', 's_drepano', 's_fever_temp', 's_fever_his',
        's_cough',  's_abdopain', 's_vomit', 's_skin', 's_earpain', 's_eyepb', 's_throat',
        's_mouthpb', 's_dysuria', 's_hematuria', 's_joint', 's_limp', 'ms_measles'
    ]
    considered_symptoms = ["symptom_" + "_".join(x.split("_")[1:]) for x in _considered_symptoms]

    df = pd.read_csv(cfg.consultation_data_path)
    df = df[_base_features + _considered_diseases + _considered_symptoms]
    df = preprocess_df(df, _base_features, base_features, _considered_diseases, considered_diseases, _considered_symptoms, considered_symptoms)

    base_features_state_dict = {
        'base_country':
        {
            'is_numeric': False,
            'vals': ['rca', 'Mali', 'Kenya', 'Tanzania', 'Niger', 'Nigeria', 'Tchad']
        },
        'base_a_age':
        {
            'is_numeric': True,
            'vals': [0, 12, 24, 48, 61]
        },
        'base_a_gender2':
        {
            'is_numeric': False,
            'vals': ['male', 'female']
        },
        'base_a_weight':
        {
            'is_numeric': True,
            'vals': [2, 5, 10, 15, 20, 30]
        },
        'base_wfa':
        {
            'is_numeric': True,
            'vals': [-3, -0.5, 0.5, 3]
        },
        'base_wfh2':
        {
            'is_numeric': True,
            'vals': [-3, -0.5, 0.5, 4]
        }
    }
    base_features_state_dict = munchify(base_features_state_dict)

    base_feature_disease_prob_dict = {}
    for disease in tqdm(considered_diseases):
        for base_feature in base_features:
            base_feature_dict = base_features_state_dict[base_feature]
            assert type(base_feature_dict.is_numeric) == bool, "'is_numeric' field should be a bool"
            states = base_feature_dict.vals
            if base_feature_dict.is_numeric:
                # num_valid = len(df[base_feature].dropna())
                prob_disease_given_base_bins = []
                pos_disease_inds = df[disease] == 1
                for i in range(len(states) - 1):
                    state_interval_inds = (df[base_feature] > states[i]) & (df[base_feature] <= states[i + 1])
                    prob_disease_given_base_bin = sum(state_interval_inds & pos_disease_inds) / sum(state_interval_inds)
                    prob_disease_given_base_bins.append(prob_disease_given_base_bin)

                base_feature_disease_prob_dict = update_dict(base_feature_disease_prob_dict, base_feature, disease, prob_disease_given_base_bins)

            elif not base_feature_dict.is_numeric:
                prob_disease_given_states = []
                # prob_disease_given_not_state = []

                pos_disease_inds = df[disease] == 1
                for state in states:
                    pos_inds = df[base_feature] == state

                    # proportion of patients with base feature that also have disease
                    # here do we compute probability given all patients that have been tested for the disease?
                    prob_disease_given_state = sum(df[pos_inds][disease] == 1) / sum(pos_inds)
                    prob_no_disease_given_state = 1 - prob_disease_given_state
                    assert (prob_disease_given_state <= 1) & (prob_no_disease_given_state <= 1), f"prob of having/ not having {disease} greater than 1"
                    prob_disease_given_states.append(prob_disease_given_state)

                    # proportion of patients without base feature that also have disease
                    # prob_disease_no_base = sum(df[neg_inds][symptom] == 1) / sum(neg_inds)
                    # prob_no_disease_no_base = 1 - prob_symptom_no_disease
                    # assert (prob_symptom_no_disease <= 1) & (prob_no_symptom_no_disease <= 1), f"prob of having/ not having {symptom} greater than 1"

                base_feature_disease_prob_dict = update_dict(base_feature_disease_prob_dict, base_feature, disease, prob_disease_given_states)

            embed()
    # filter symptoms/diseases that have less than a certain amount of valid entries
    df_diseases = df[considered_diseases]
    df_symptoms = df[considered_symptoms]
    fever_bins = [35, 37, 40, 44]

    # here would need to figure out way to automatically infer RV type and extract probabilities accordingly
    disease_symptom_prob_dict = {}
    for symptom in tqdm(considered_symptoms):
        for disease in considered_diseases:
            # condition here should really be 'if disease variable is binary'
            if not symptom == "symptom_temp":
                # print(disease)
                # extract positive and negative cases, determine probability of disease based on that
                vals, inds = np.unique(df[disease], return_inverse=True)
                neg_inds = inds == 0
                pos_inds = inds == 1
                num_valid = sum(neg_inds) + sum(pos_inds)
                assert np.all(np.array(df[neg_inds][disease]) == 0), "not all negative indices correspond to negative cases"
                assert np.all(np.array(df[pos_inds][disease]) == 1), "not all negative indices correspond to negative cases"

                # proportion of patients with disease that also have symptom
                prob_symptom_with_disease = sum(df[pos_inds][symptom] == 1) / sum(pos_inds)
                prob_no_symptom_with_disease = 1 - prob_symptom_with_disease
                assert (prob_symptom_with_disease <= 1) & (prob_no_symptom_with_disease <= 1), f"prob of having/ not having {symptom} greater than 1"

                # proportion of patients without disease that also have symptom
                prob_symptom_no_disease = sum(df[neg_inds][symptom] == 1) / sum(neg_inds)
                prob_no_symptom_no_disease = 1 - prob_symptom_no_disease
                assert (prob_symptom_no_disease <= 1) & (prob_no_symptom_no_disease <= 1), f"prob of having/ not having {symptom} greater than 1"

                disease_symptom_prob_dict = update_dict(disease_symptom_prob_dict, disease, symptom, prob_symptom_with_disease)
                if symptom in disease_symptom_prob_dict.keys():
                    if disease_symptom_prob_dict[symptom]:
                        disease_symptom_prob_dict[symptom][disease] = np.array([prob_symptom_no_disease, prob_symptom_with_disease])
                    else:
                        disease_symptom_prob_dict[symptom] = {}
                        disease_symptom_prob_dict[symptom][disease] = np.array([prob_symptom_no_disease, prob_symptom_with_disease])
                else:
                    disease_symptom_prob_dict[symptom] = {}
                    disease_symptom_prob_dict[symptom][disease] = np.array([prob_symptom_no_disease, prob_symptom_with_disease])

                # disease_symptom_prob_dict[symptom] = {disease: np.array([prob_symptom_with_disease, prob_symptom_no_disease])}
            else:
                num_valid = len(df[disease].dropna())
                num_pos = sum(df[disease] == 1)
                prob_symptom_within_fever_range = []
                for i in range(len(fever_bins) - 1):
                    fever_inds = (df[symptom] > fever_bins[i]) & (df[symptom] <= fever_bins[i + 1])
                    pos_disease_inds = df[disease] == 1
                    p = sum(df[fever_inds & pos_disease_inds][disease] == 1) / num_pos
                    prob_symptom_within_fever_range.append(p)

                if symptom in disease_symptom_prob_dict.keys():
                    if disease_symptom_prob_dict[symptom]:
                        disease_symptom_prob_dict[symptom][disease] = np.array(prob_symptom_within_fever_range)
                    else:
                        disease_symptom_prob_dict[symptom] = {}
                        disease_symptom_prob_dict[symptom][disease] = np.array(prob_symptom_within_fever_range)
                else:
                    disease_symptom_prob_dict[symptom] = {}
                    disease_symptom_prob_dict[symptom][disease] = np.array(prob_symptom_within_fever_range)

    metadata_dict = {
        'base_disease': base_feature_disease_prob_dict,
        'disease_symptom': disease_symptom_prob_dict
    }

    pickle_object(metadata_dict, cfg.consultation_data_prob_dict_path)

    embed()
