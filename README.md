# PhantoMedicus - Medical Survey Generator
Phantomedicus is an early stage framework for simulating patients and consultations. Two methods are currently supported:
- Manually assigned probabilities
- Data driven probabilities

Either of these methods can be run by changing a CLI: `python main.py --bayes manual_probs` is used to generate a simulator given manually designated probabilities, an example of which can be found in `metadata.json`, and `python main.py --bayes data_driven_probs` makes use of an already existing dataset to derive the probabilistic interdepencies between different base attributes, diseases, and symptoms. To create the environment run `conda env create -f environment.yml`. 

##  Bayesian Network Structure
The graph dependencies can be broadly summarized as base features influencing the likelihood of certain diseases, which in turn influence a patient's symptoms.
The approach for defining the structure and corresponding probabilities is outlined below.

## Manual Probabilities
The metadata structure which is currently used is a dictionary of the following form:

```
metadata_dict = {
    "disease_list": considered_diseases,
    "symptom_list": considered_symptoms,
    "node_states": {
        "patient_attributes": base_features_state_dict,
        "diseases": disease_state_dict,
        "symptoms": symptom_state_dict,
    },
    "patient_attribute_disease_probs": base_feature_disease_prob_dict,
    "disease_symptom_probs": disease_symptom_prob_dict,
    "doctors": doctors,
}
```

- `disease_list` contains the list of diseases that you wish to include in your model, all prefixed by `disease` e.g. `disease_pneumonia`
- `symptom_list` contains the list of symptoms that you wish to include in your model, all prefixed by `symptom` e.g. `symptom_pneumonia`
- `node_states` contains descriptive features for the random variables (nodes) in the graph. Note that these vary between the patient attributes and symptoms/ diseases as we do not assign marginal probabilities to the symptoms/ diseases. For this we need to define a structure of probabilistic dependencies as outlined below. This has three subdictionaries:
    - `patient_attributes` - here we have 4 key-value pairs:
        - `dtype` i.e. the datatype, can be one of `binary`, `categorical`, or `continuous`
        - `state_name` i.e. the names the random variable may assume
        - `vals` i.e. the values assumed for each of the states (often just the state names themselves)  
        - `prob` i.e. the probability of sampling any one of these states
    - `diseases` - here we have 2 key-value pairs
       - `dtype` as described above
       - `state_name` as described above
    - `symptoms` - here we also have 2 key-value pairs
       - `dtype` as described above
       - `state_name` as described above
- `patient_attribute_disease_probs` - here, for each patient attribute we define a subdictionary. Each subdictionary will contain 
  the diseases which are influenced by each patient attribute (i.e. edges in the Bayesian network), alongside the associated probabilities of the diseases due to 
  each possible state of each given patient attribute. For instance if we have a patient attribute `base_country` for which 
  4 possible states i.e. countries are assigned, we may define the subdictionary corresponding the `base_country` as follows:
  ```
    "base_country": {
            "disease_urti": [0.07, 0.04, 0.05, 0.04], 
            "disease_bronchiolitis": [0.07, 0.04, 0.05, 0.04], 
            "disease_bronchitis": [0.07, 0.04, 0.05, 0.04],
            "disease_pneumonia": [0.07, 0.04, 0.05, 0.04], 
            "disease_asthma": [0.07, 0.04, 0.05, 0.04], 
            "disease_tb": [0.07, 0.04, 0.05, 0.04], 
            "disease_covid": [0.07, 0.04, 0.05, 0.04], 
            "disease_malaria": [0.07, 0.04, 0.05, 0.04], 
            "disease_dengue": [0.07, 0.04, 0.05, 0.04], 
            "disease_diarrhea": [0.07, 0.04, 0.05, 0.04], 
            "disease_ebola": [0.07, 0.04, 0.05, 0.04], 
            "disease_severe": [0.07, 0.04, 0.05, 0.04]
        },
  ```
- `disease_symptom_probs` is much the same as `patient_attribute_disease_probs` except we now define the associated probabilities
  of symptoms based on diseases.
- `doctors` contains a subdictionary with the following fields:
    - `doctor_types` - list of the names associated with the doctor types and can be found in `config.py`
    - `country` contains a further subdictionary with all the countries you are simulating. For each country we assign a probability
      distribution of the doctor profiles, as well as doctor specific parameters for each doctor (serves to simulate differences 
      in doctors across different regions)
      
A comprehensive example of the above can be found in `metadata.json`, which is a metadata file with manually assigned probabilities.
    
## Data Driven
The data driven approach makes use of the same metadata structure as above, the only difference being that now the probabilities are 
derived from a dataset. The procedure can be found in `generate_prob_dict.py`. Note that if another dataset is used, it will 
require some modifications to pick the specific patient attributes/ diseases/ symptoms of interest.

##  Doctor Profiles for Consultations
The defined doctor profiles can be found in `src/doctor.py`. Note that the doctor profiles are used in `main.py` when simulating 
patients and conducting consultations.

## Repository Structure
- `src/doctor.py` contains the defined doctor profiles
- `src/patient_simulator.py` contains the `PatientSimulator` class which defines the Bayesian network structure and aggregates the probabilities 
  using the metadata described above
- `src/utils.py` contains utility functions for manipulating patient data and for the doctor profiles
- `config.py` contains some configuration parameters for the simulation and paths for reading/outputting data
- `generate_prob_dict.py` - contains the code for generating the metadata based on the raw data
- `main.py` contains the entire procedure for simulating batches of patients and their consultations and outputs the consultations
in a `pkl` file 
