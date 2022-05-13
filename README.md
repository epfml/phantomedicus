# PhantoMedicus - Medical Survey Generator
Phantomedicus is an early stage framework for simulating patients and consultations. Two methods are currently supported:
- Manually assigned probabilities
- Data driven probabilities

Either of these methods can be run by changing a CLI: `python main.py manual_probs` is used to generate a simulator given manually designated probabilities, an example of which can be found in `metadata.json`, and `python main.py data_drive_probs` makes use of an already existing dataset to derive the probabilistic interdepencies between different base attributes, diseases, and symptoms.

# Bayesian Network Structure
The graph dependencies can be broadly summarized as base features influencing the likelihood of certain diseases, which in turn influence a patient's symptoms.

## Manual Probabilities
## Data Driven

# Doctor Profiles for Consultations
