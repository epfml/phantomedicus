"""Script for preprocessing existing consultation data to generate Bayesian Network"""
import json
import pickle
import numpy as np
import pandas as pd
from IPython import embed
from munch import munchify

from config import load_config
from src.utils import pickle_object

# TODO: automate parsing based on datatypes e.g. binning for continuous values
# TODO: test out parameter learning functionality of pgmpy to learn CPDs based on a predefined probabilistic DAG
# -> https://pgmpy.org/examples/Learning%20Parameters%20in%20Discrete%20Bayesian%20Networks.html?highlight=parameter%20learning


def extract_column_data_type(dict_csv):
    pass


def parameter_learning(PDAG, filtered_df):
    pass


if __name__=="__main__":
    with open("metadata.json") as f:
        metadata = json.load(f)
    metadata = munchify(metadata)
    cfg = load_config(metadata)

    _base_features = []
    base_features = []

    _considered_diseases = [
        'classify_uti_febrile', 'classify_abdo_constipation', 'classify_abdo_gastro',
        'classify_abdo_resp_pain', 'classify_abdo_unclassified', 'classify_anaphylaxis',
        'classify_anaphylaxis_severe', 'classify_anemia_non_severe', 'classify_anemia_severe',
        'classify_cough_bronchiolitis', 'classify_cough_persistent', 'classify_cough_pneumonia',
        'classify_cough_urti', 'classify_cystitis_acute', 'classify_dehydration_moderate',
        'classify_dehydration_none', 'classify_diarrhoea_persistent', 'classify_dysentery_non_compl',
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
        's_temp', 's_oedema', 's_drepano', 's_fever_temp', 's_fever_his',
        's_cough', 's_diarr', 's_abdopain', 's_vomit', 's_skin', 's_earpain', 's_eyepb', 's_throat',
        's_mouthpb', 's_dysuria', 's_hematuria', 's_joint', 's_limp', 'ms_measles'
    ]
    considered_symptoms = ["symptom_" + "_".join(x.split("_")[1:]) for x in _considered_symptoms]


    df = pd.read_csv(cfg.consultation_data_path)
    df = df[_base_features + _considered_diseases + _considered_symptoms]
    prev_names = _base_features + _considered_diseases + _considered_symptoms
    new_names = base_features + considered_diseases + considered_symptoms
    df.rename(columns={prev: new for prev, new in zip(prev_names, new_names)}, inplace=True)

    replace_dict = {'none': np.nan}
    df['symptom_temp'].replace(',', '.', regex=True, inplace=True)
    df.replace(replace_dict, inplace=True)
    df = df.astype(np.float32)

    # filter symptoms/diseases that have less than a certain amount of valid entries
    df_diseases = df[considered_diseases]

    df_symptoms = df[considered_symptoms]
    fever_bins = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]

    # here would need to figure out way to automatically infer RV type and extract probabilities accordingly
    disease_symptom_prob_dict = {}
    for symptom in considered_symptoms:
        for disease in considered_diseases:
            # condition here should really be 'if disease variable is binary'
            if not symptom == "symptom_temp":
                print(disease)
                # extract positive and negative cases, determine probability of disease based on that
                vals, inds = np.unique(df[disease], return_inverse=True)
                neg_inds = inds == 0
                pos_inds = inds == 1
                num_valid = neg_inds.shape[0] + pos_inds.shape[0]

                assert np.all(np.array(df[neg_inds][disease]) == 0), "not all negative indices correspond to negative cases"
                assert np.all(np.array(df[pos_inds][disease]) == 1), "not all negative indices correspond to negative cases"

                # proportion of patients with disease that also have symptom
                prob_symptom_with_disease = sum(df[pos_inds][symptom] == 1) / num_valid
                prob_no_symptom_with_disease = 1 - prob_symptom_with_disease
                assert (prob_symptom_with_disease <= 1) & (prob_no_symptom_with_disease <= 1), f"prob of having/ not having {symptom} greater than 1"

                # proportion of patients without disease that also have symptom
                prob_symptom_no_disease = sum(df[neg_inds][symptom] == 1) / num_valid
                prob_no_symptom_no_disease = 1 - prob_symptom_no_disease
                assert (prob_symptom_no_disease <= 1) & (prob_no_symptom_no_disease <= 1), f"prob of having/ not having {symptom} greater than 1"

                if symptom in disease_symptom_prob_dict.keys():
                    if disease_symptom_prob_dict[symptom]:
                        disease_symptom_prob_dict[symptom][disease] = np.array([prob_symptom_with_disease, prob_symptom_no_disease])
                    else:
                        disease_symptom_prob_dict[symptom] = {}
                        disease_symptom_prob_dict[symptom][disease] = np.array([prob_symptom_with_disease, prob_symptom_no_disease])
                else:
                    disease_symptom_prob_dict[symptom] = {}
                    disease_symptom_prob_dict[symptom][disease] = np.array([prob_symptom_with_disease, prob_symptom_no_disease])

                # disease_symptom_prob_dict[symptom] = {disease: np.array([prob_symptom_with_disease, prob_symptom_no_disease])}
            else:
                num_valid = len(df[disease].dropna())
                prob_symptom_within_fever_range = []
                for i in range(len(fever_bins) - 1):
                    fever_inds = (df[symptom] > fever_bins[i]) & (df[symptom] <= fever_bins[i + 1])
                    pos_disease_inds = df[disease] == 1
                    p = sum(df[fever_inds & pos_disease_inds][disease] == 1) / num_valid
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

    embed()

    pickle_object(disease_symptom_prob_dict, cfg.consultation_data_prob_dict_path)

