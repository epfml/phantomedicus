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


class Patient:
    def __init__(self, metadata, gender, age, country):
        # define symptoms of patient
        self.gender = gender
        self.age = age
        self.country = country
        self.metadata = metadata
        self.assign_diagnoses()

    def assign_diagnoses(self):
        dx_states = self.metadata.node_states.dx
        self.diagnoses = {dx: np.random.choice(dx_states[dx].state_names, 1, p=dx_states[dx].prob)[0] for dx in self.metadata.diagnosis_list}


class PatientSimulator:
    # TODO: define probability matrix (DAG) associating symptoms/ environmental factors with diseases
    # TODO: generate patient profile randomly based on predefined ranges of values/ lists of  (location, age, ethnicity, etc.)
    # TODO: use probability matrix and patient profiles to determine if/ which disease(s) is/ are present
    def __init__(self, metadata):
        self.metadata = metadata
        self.init_bayesian_model()
        self.symptoms = self.metadata.symptom_list
        self.diagnoses = self.metadata.diagnosis_list


    def init_bayesian_model(self):
        # Defining the model structure by passing edge list of dx->symp dependencies
        edge_list = [(x, y) for x in self.metadata.symp_dx_prob.keys() for y in self.metadata.symp_dx_prob[x]]
        self.model = BayesianModel(edge_list)

        # adding individual cpd states for diagnoses
        for dx, state in self.metadata.node_states.dx.items():
            cpd = TabularCPD(variable=dx, variable_card=len(state.prob), values=[[x] for x in state.prob], state_names={dx: state.state_names})
            self.model.add_cpds(cpd)

        # adding cpd states for symptoms based on diagnoses
        for symp in self.metadata.symptom_list:
            associated_dx = list(self.model.predecessors(symp))
            # values = [[0] * 2 ** len(associated_dx) for _ in range(len(self.metadata.node_states.symp[symp].state_names))]
            # values = np.zeros((len(self.metadata.node_states.symp[symp].state_names), 2 ** len(associated_dx)))

            # associated_dx_state_combinations = list(product(*[self.metadata.node_states.dx[dx].state_names for dx in associated_dx]))
            associated_dx_state_combinations = np.array(list(product(*[[1, 0] for dx in associated_dx])))
            associated_dx_probs = np.array([self.metadata.symp_dx_prob[dx][symp] for dx in associated_dx])

            # defining probability of symptom for each combination of diagnoses - simple max for now
            # aggregating individual cpds into multimodal cpds: https://www.cmu.edu/dietrich/sds/ddmlab/papers/GonzalezVrbin2007.pdf
            prob_dist_combinations = associated_dx_probs * associated_dx_state_combinations
            max_prob_per_comb = prob_dist_combinations.max(1).reshape(1, -1)
            values = np.concatenate([max_prob_per_comb, 1 - max_prob_per_comb], axis=0)

            symp_state_names = {symp: self.metadata.node_states.symp[symp].state_names}
            dx_state_names = {dx: self.metadata.node_states.dx[dx].state_names for dx in associated_dx}
            cpd = TabularCPD(variable=symp,
                             values=values,
                             state_names=symp_state_names | dx_state_names,
                             variable_card=len(self.metadata.node_states.symp[symp].state_names),
                             evidence=associated_dx, evidence_card=[len(self.metadata.node_states.dx[dx].state_names) for dx in associated_dx])

            self.model.add_cpds(cpd)

        print("Bayesian network configured\n")

    def inference(self, patient):
        infer = VariableElimination(self.model)
        symptom_probs = infer.query(self.symptoms, evidence=patient.diagnoses)

        return symptom_probs.sample(1)

class SimulateSurvey:
    # TODO: incorporate missingness in survey questions randomly (perhaps conditionally i.e. depending on doctor/ region)
    # TODO: return list of (q, a) tuples per patient
    def __init__(self, bayes_net, n_patients, metadata):
        self.bayes_net = bayes_net
        self.n_patients = n_patients
        self.metadata = metadata
        self.survey = pd.DataFrame(columns=[*self.metadata.symptom_list, *self.metadata.diagnosis_list])

    def run_simulation(self):
        for i in range(self.n_patients):
            patient = Patient(self.metadata, gender="female", age=20, country="switzerland")
            diagnoses = pd.Series(patient.diagnoses)
            symptoms = self.bayes_net.inference(patient).squeeze()
            row = pd.concat([symptoms, diagnoses])
            self.survey = self.survey.append(row, ignore_index=True)

        return self.survey

    def save_survey(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.survey.to_csv(path + "/patient_data.csv", mode='w', index=False)
        print(f"Survey saved to {path}/patiend_data.csv")


if __name__=="__main__":
    with open("metadata.json") as f:
        metadata = json.load(f)
    metadata = munchify(metadata)

    patient = Patient(metadata, gender="female", age=20, country="switzerland")
    patientSimulator = PatientSimulator(metadata)
    surveySimulator = SimulateSurvey(patientSimulator, 1000, metadata)
    df_survey = surveySimulator.run_simulation()
    surveySimulator.save_survey("data")

    embed()

