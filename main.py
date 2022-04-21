"""Run end-to-end simulation comprising patient generation and doctor consultation"""
import json
import pandas as pd
from IPython import embed
from munch import munchify
from pgmpy.factors.discrete import State

from src.doctor import *
from config import load_config
from src.utils import pickle_object
from src.patient_simulator import PatientSimulator


if __name__ == "__main__":
    # data = pd.read_csv("data/patient_data.csv")
    with open("metadata.json") as f:
        metadata = json.load(f)
    metadata = munchify(metadata)

    cfg = load_config(metadata)
    PatientBayesNet = PatientSimulator(metadata)

    consultation_collection = []
    for clinic in range(cfg.n_clinics):
        country = np.random.choice(metadata.node_states.patient_attributes.country.state_names)

        evidence = [State('country', country)]
        df_patients = PatientBayesNet.run_simulation(cfg.n_patients_per_clinic, evidence=evidence)
        patient_list = PatientBayesNet.df_to_patient_batch(df_patients)

        doc_type = np.random.choice(metadata.doctors.types, p=metadata.doctors.prob_by_country[country])
        print(doc_type)
        if doc_type == "decision_tree":
            cfg.doctor_kwargs['data'] = df_patients
        doctor = cfg.doctor_dict[doc_type](**cfg.doctor_kwargs)

        clinic_consultations = []
        for _, patient in df_patients.iterrows():
            clinic_consultations.append(doctor.conduct_consultation(patient))

        consultation_collection.append(clinic_consultations)

    pickle_object(consultation_collection, "data/consultation_collections.pkl")

