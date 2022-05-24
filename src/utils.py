"""Utility functions for simulating patients and consultations"""
import pickle
import graphviz
import numpy as np
from sklearn import tree
from IPython import embed


def one_hot_encode_simulated_patient_df(df, metadata):
    df.replace({"True": 1, "False": 0}, inplace=True)

    for base_attr in metadata.node_states["patient_attributes"].keys():
        if not metadata.node_states["patient_attributes"][base_attr].dtype == "binary":
            unique_col_entries = np.unique(df[base_attr])
            for entry in unique_col_entries:
                df["_".join([base_attr, entry])] = np.where(
                    df[base_attr] == entry, 1, 0
                )
            df.drop(base_attr, axis=1, inplace=True)

    for symptom in metadata.node_states["symptoms"].keys():
        if metadata.node_states["symptoms"][symptom].dtype == "continuous":
            unique_col_entries = np.unique(df[symptom])
            for entry in unique_col_entries:
                df["_".join([symptom, entry])] = np.where(df[symptom] == entry, 1, 0)
            df.drop(symptom, axis=1, inplace=True)

    return df


def enumerate_categorical_variables(metadata):
    enumerate_dict = {}
    for key in metadata.node_states.keys():
        nodes = metadata.node_states[key]
        for node in nodes.keys():
            if nodes[node].dtype in ["categorical", "continuous"]:
                enumerate_dict[node] = {
                    k: v for k, v in enumerate(nodes[node].state_names)
                }

    return enumerate_dict


def reverse_dict(_dict):
    rev_dict = {}
    for key, val in _dict.items():
        rev_dict[key] = {v: k for k, v in val.items()}

    return rev_dict


def decision_tree_consultation(clf, patient, features, categorical_mapping):
    """
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    """
    x = patient.values.reshape(1, -1)
    # x = np.where(x == "True", 1, 0)
    leaf_id = clf.apply(x)
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_indicator = clf.decision_path(x)

    # node_index = node_indicator.indices[node_indicator.indptr[i] : node_indicator.indptr[i + 1]]
    node_index = node_indicator.indices[:-1]
    # assert clf.tree_.max_depth == node_index.shape[0], "tree depth and consulation path mismatch..." # this assertion doesn't have to be true..
    questions = np.array(features)[feature[node_index]]
    answers = x[:, feature[node_index]].reshape(-1)
    str_answers = [
        categorical_mapping[q][a] if q in categorical_mapping else ["False", "True"][a]
        for q, a in zip(questions, answers)
    ]

    assert all(
        patient[questions] == answers
    ), "consultation questions don't match up with patient symptoms"

    return [(q, a) for q, a in zip(questions, str_answers)]


def plot_decision_tree(clf, data_dict, file_path):
    """
    https://scikit-learn.org/stable/modules/tree.html#classification
    """
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=data_dict["feature_names"],
        class_names=data_dict["target_names"],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(f"{file_path}")

    return graph


def pickle_object(obj, path):
    with open(f"{path}", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Binary file saved to {path}")
