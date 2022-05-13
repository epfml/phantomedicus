"""Run end-to-end simulation comprising patient generation and doctor consultation"""
import sys
import json
import pandas as pd
from IPython import embed
from munch import munchify
from pgmpy.factors.discrete import State

from src.doctor import *
from config import load_config
from src.utils import pickle_object, one_hot_encode_patient_df
from src.patient_simulator import PatientSimulator


if __name__ == "__main__":
    # data = pd.read_csv("data/patient_data.csv")
    assert len(sys.argv) == 2, "Please input the BayesNet generation method ('manual_probs' or 'data_driven_probs')"

    if sys.argv[1] == "manual_probs":
        with open("metadata.json") as f:
            metadata = json.load(f)
    elif sys.argv[1] == "data_driven_probs":
        with open("data/consultation_data_prob_dict.pkl", "rb") as handle:
            metadata = pickle.load(handle)
    else:
        print("Please input a valid BayesNet generation method ('manual_probs', 'data_driven_probs')\n")
        quit()

    metadata = munchify(metadata)
    cfg = load_config(metadata)
    BayesNet = PatientSimulator(metadata)
    # df_trial = BayesNet.run_simulation(100000)

    consultation_collection = {}
    for clinic in range(cfg.n_clinics):
        country = np.random.choice(metadata.node_states.patient_attributes.base_country.state_names)

        evidence = [State('base_country', country)]
        df_patients = BayesNet.run_simulation(cfg.n_patients_per_clinic, evidence=evidence)
        # patient_list = BayesNet.df_to_patient_batch(df_patients)
        df_patients = one_hot_encode_patient_df(df_patients, metadata)
        metadata['data'] = df_patients

        doc_type = np.random.choice(metadata.doctors.types, p=metadata.doctors.country[country].prob_doctor)
        print(doc_type)
        if doc_type == "decision_tree":
            cfg.doctor_kwargs['data'] = df_patients
        # TODO: Set proper doctor kwargs here!!
        doctor = cfg.doctor_dict[doc_type](**metadata)

        clinic_consultations = []
        for _, patient in df_patients.iterrows():
            clinic_consultations.append(doctor.conduct_consultation(patient))

        consultation_collection["_".join([country, clinic])] =  clinic_consultations

    pickle_object(consultation_collection, "data/consultation_collections.pkl")

