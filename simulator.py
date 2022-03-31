"""Simulating clinical data probabilistically"""
# TODO: read up on synthetic clinical data simulators
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from IPython import embed
from munch import munchify
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling


class Patient:
    def __init__(self, metadata, gender, age, country):
        # define symptoms of patient
        self.gender = gender
        self.age = age
        self.country = country
        self.metadata = metadata
        self.assign_diagnoses()

    def assign_diagnoses(self):
        disease_states = self.metadata.node_states.diseases
        self.diagnoses = {disease: np.random.choice(disease_states[disease].state_names, 1, p=disease_states[disease].prob)[0] for disease in self.metadata.disease_list}


class PatientSimulator:
    # TODO: define probability matrix (DAG) associating symptoms/ environmental factors with diseases
    # TODO: generate patient profile randomly based on predefined ranges of values/ lists of  (location, age, ethnicity, etc.)
    # TODO: use probability matrix and patient profiles to determine if/ which disease(s) is/ are present
    def __init__(self, metadata):
        self.metadata = metadata
        self.init_bayesian_model()
        self.symptoms = self.metadata.symptom_list
        self.diagnoses = self.metadata.disease_list


    def init_bayesian_model(self):
        # Defining the model structure by passing edge list of disease->symp dependencies
        base_attr_disease_edge_list = [(x, y) for x in self.metadata.patient_attribute_disease_probs.keys() for y in self.metadata.patient_attribute_disease_probs[x]]
        disease_symptom_edge_list = [(x, y) for x in self.metadata.disease_symptom_probs.keys() for y in self.metadata.disease_symptom_probs[x]]
        self.model = BayesianModel(base_attr_disease_edge_list + disease_symptom_edge_list)

        # adding base patient attributes
        for base_attr, state in self.metadata.node_states.patient_attributes.items():
            cpd = TabularCPD(variable=base_attr, variable_card=len(state.prob), values=[[x] for x in state.prob], state_names={base_attr: state.state_names})
            self.model.add_cpds(cpd)

        # adding cpd states and probabilities for diagnoses based on base attributes
        for disease in self.metadata.disease_list:
            associated_base_attr = list(self.model.predecessors(disease))
            base_state_combs = np.array(list(product(*[np.arange(len(self.metadata.node_states.patient_attributes[base_attr].prob)) for base_attr in associated_base_attr]))) # [::-1]
            base_attr_probs = np.array([self.metadata.patient_attribute_disease_probs[base_attr][disease] for base_attr in associated_base_attr])
            prob_dist_combinations = base_attr_probs[np.arange(len(associated_base_attr)), base_state_combs]
            max_prob_per_comb = prob_dist_combinations.max(1).reshape(1, -1)
            values = np.concatenate([max_prob_per_comb, 1 - max_prob_per_comb], axis=0) # currently this method only works for binary target variables

            disease_state_names = {disease: self.metadata.node_states.diseases[disease].state_names}
            base_attr_state_names = {base_attr: self.metadata.node_states.patient_attributes[base_attr].state_names for base_attr in associated_base_attr}

            cpd = TabularCPD(variable=disease,
                             values=values,
                             state_names=disease_state_names | base_attr_state_names,
                             variable_card=len(self.metadata.node_states.diseases[disease].state_names),
                             evidence=associated_base_attr, evidence_card=[len(self.metadata.node_states.patient_attributes[base_attr].state_names) for base_attr in associated_base_attr]
                             )
            self.model.add_cpds(cpd)

        # adding cpd states for symptoms based on diagnoses
        for symp in self.metadata.symptom_list:
            associated_disease = list(self.model.predecessors(symp))

            associated_disease_state_combinations = np.array(list(product(*[[1, 0] for disease in associated_disease])))
            associated_disease_probs = np.array([self.metadata.disease_symptom_probs[disease][symp] for disease in associated_disease])

            # defining probability of symptom for each combination of diagnoses - simple max for now
            # aggregating individual cpds into multivariate cpds: https://www.cmu.edu/dietrich/sds/ddmlab/papers/GonzalezVrbin2007.pdf
            prob_dist_combinations = associated_disease_probs * associated_disease_state_combinations
            max_prob_per_comb = prob_dist_combinations.max(1).reshape(1, -1)
            max_prob_per_comb = np.where(max_prob_per_comb == 0, np.min(max_prob_per_comb[max_prob_per_comb.nonzero()]), max_prob_per_comb)
            assert np.min(max_prob_per_comb) > 0, f"there is a disease configuration for which the probability of {symp} is 0"

            values = np.concatenate([max_prob_per_comb, 1 - max_prob_per_comb], axis=0)

            symp_state_names = {symp: self.metadata.node_states.symptoms[symp].state_names}
            disease_state_names = {disease: self.metadata.node_states.diseases[disease].state_names for disease in associated_disease}
            cpd = TabularCPD(variable=symp,
                             values=values,
                             state_names=symp_state_names | disease_state_names,
                             variable_card=len(self.metadata.node_states.symptoms[symp].state_names),
                             evidence=associated_disease, evidence_card=[len(self.metadata.node_states.diseases[disease].state_names) for disease in associated_disease])

            self.model.add_cpds(cpd)

        print("Bayesian network configured\n")

    def inference(self):
        # infer = VariableElimination(self.model) # querying very slow with additional base factors
        # symptom_probs = infer.query(self.symptoms, evidence=patient.diagnoses)
        inference = BayesianModelSampling(self.model) # instead we opt for

        return inference


class SimulateSurvey:
    # TODO: incorporate missingness in survey questions randomly (perhaps conditionally i.e. depending on doctor/ region)
    # TODO: return list of (q, a) tuples per patient
    def __init__(self, bayes_net, n_patients, metadata):
        self.bayes_net = bayes_net
        self.n_patients = n_patients
        self.metadata = metadata

    def run_simulation(self):
        inference_engine = self.bayes_net.inference()
        self.survey = inference_engine.forward_sample(self.n_patients)

        print("Dataset generated")
        return self.survey

    def save_survey(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.survey.to_csv(path + "/patient_data.csv", mode='w', index=False)
        print(f"Survey saved to {path}/patient_data.csv")


if __name__=="__main__":
    with open("metadata.json") as f:
        metadata = json.load(f)
    metadata = munchify(metadata)

    bayes_net = PatientSimulator(metadata)
    surveySimulator = SimulateSurvey(bayes_net, 100000, metadata)
    df_survey = surveySimulator.run_simulation()
    surveySimulator.save_survey("data")

    embed()

