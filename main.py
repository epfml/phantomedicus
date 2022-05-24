"""Run end-to-end simulation comprising patient generation and doctor consultation"""
import sys
import json
import pandas as pd
from IPython import embed
from munch import munchify
from pgmpy.factors.discrete import State

from src.doctor import *
from config import load_config
from options import parse_args
from src.patient_simulator import PatientSimulator
from src.utils import pickle_object, one_hot_encode_simulated_patient_df, enumerate_categorical_variables, reverse_dict


if __name__ == "__main__":
    args = parse_args()

    if args.bayes == "manual_probs":
        with open("metadata.json") as f:
            metadata = json.load(f)
    elif args.bayes == "data_driven_probs":
        with open("data/consultation_data_prob_dict.pkl", "rb") as handle:
            metadata = pickle.load(handle)

    metadata = munchify(metadata)
    enumerate_dict = enumerate_categorical_variables(metadata)

    cfg = load_config(metadata)
    BayesNet = PatientSimulator(metadata)
    df_trial = BayesNet.run_simulation(50000, reject_diseaseless_patients=True)

    consultation_collection = {}
    for clinic in range(cfg.n_clinics):
        country = np.random.choice(metadata.node_states.patient_attributes.base_country.state_names)
        evidence = [State('base_country', country)]
        df_patients = BayesNet.run_simulation(cfg.n_patients_per_clinic, evidence=evidence)
        # patient_list = BayesNet.df_to_patient_batch(df_patients)
        # df_patients = one_hot_encode_simulated_patient_df(df_patients, metadata)
        for k, v in reverse_dict(enumerate_dict).items():
            df_patients[k].replace(v, inplace=True)
        df_patients.replace({"True": 1, "False": 0}, inplace=True)
        metadata['data'] = df_patients

        doc_type = np.random.choice(metadata.doctors.types, p=metadata.doctors.country[country].prob_doctor)

        consultation_kwargs = {
            "symptom_list": metadata.symptom_list,
            "disease_list": metadata.disease_list,
            "base_feature_list": list(metadata.node_states.patient_attributes.keys()),
            "doc": metadata.doctors.country[country].doctor_kwargs[doc_type],
            "data": df_patients,
            "enumerate_dict": enumerate_dict
        }

        print(doc_type)
        doctor = cfg.doctor_dict[doc_type](**consultation_kwargs)

        clinic_consultations = []
        for _, patient in df_patients.iterrows():
            clinic_consultations.append(doctor.conduct_consultation(patient))
        consultation_collection["_".join([country, str(clinic)])] =  clinic_consultations

    pickle_object(consultation_collection, "data/consultation_collections.pkl")

