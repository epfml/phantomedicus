"""Utility functions for simulating patients and consultations"""
import graphviz
from sklearn import tree
from IPython import embed


def decision_tree_consultation(clf, patient, symptoms):
    """
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    """
    x = patient.values.reshape(1, -1)
    leaf_id = clf.apply(x)
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_indicator = clf.decision_path(x)

    # node_index = node_indicator.indices[node_indicator.indptr[i] : node_indicator.indptr[i + 1]]
    node_index = node_indicator.indices[:-1]
    # assert clf.tree_.max_depth == node_index.shape[0], "tree depth and consulation path mismatch..." # this assertion doesn't have to be true..
    questions = symptoms[feature[node_index]]
    answers = x[:, feature[node_index]]

    return [(q, a) for q, a in zip(questions, answers)]


def plot_decision_tree(clf, data_dict, file_path):
    dot_data = tree.export_graphviz(clf, out_file=None,
                          feature_names=data_dict['feature_names'],
                          class_names=data_dict['target_names'],
                          filled=True, rounded=True,
                          special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"{file_path}")

    return graph


