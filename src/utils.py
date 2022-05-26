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
    x = patient[features].values.reshape(1, -1)
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


def extract_leaf_node_ids(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return is_leaves


def dt_reverse_order_consultation(clf, patient, features, targets, leaf_target, min_correct_ans, categorical_mapping):
    """
    We want to traverse the decision tree in reverse order given a prior assumption on the
    target disease, whilst still guaranteeing a minimum % of correct answers
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    """
    string_data = tree.export_graphviz(clf, out_file=None)
    string_data = string_data.split("\n")[3:]

    # define splitting conditions at each node
    node_data = [string_data[0]] + string_data[1:-1][::2]
    node_data_dict = {
        int(k): v for line in node_data for k, v in [(line.split(" ")[0], "".join(line.split(" ")[1:]))]
    }

    # define node predecessor dictionary
    dt_edges = string_data[2::2]
    dt_edges_dict = {
        v: k for line in dt_edges for k, v in [(int(x) for i, x in enumerate(line.split(" ")) if i in [0, 2])]
    }

    is_leaves = extract_leaf_node_ids(clf)
    leaf_ids = np.argwhere(is_leaves==True)
    leaf_preds = clf.tree_.value[leaf_ids]
    leaf_preds = leaf_preds.argmax(-1).reshape(-1, len(targets))

    # np.unique(leaf_preds.sum(1), return_counts=True) # interesting to see how many leaves predict multiple diseases
    assert leaf_preds.shape[0] == clf.get_n_leaves(), "mismatch in number of leaves"

    # currently only consider first present disease. Should be extended to multiple diseases but would require
    # additional logic to account for edge cases
    leaf_target = targets.index(leaf_target)
    target_leaves = np.argwhere(leaf_preds.argmax(1) == leaf_target)

    # try and obtain a dt path of questions corresponding to disease of choice with min % of correct answers
    found_valid_traversal = False
    children_left = clf.tree_.children_left
    for target_leaf in target_leaves:
        target_leaf_id = leaf_ids[target_leaf].item()

        curr_id = target_leaf_id
        dt_path = [curr_id]

        while not curr_id == 0:
            parent_node = dt_edges_dict[curr_id]
            dt_path.insert(0, parent_node)
            curr_id = parent_node

        # check if left or right child to know if bigger or smaller
        consultation_dict = {}
        for i, node in enumerate(dt_path[:-1]):
            feature, thresh = node_data_dict[node].split("\\")[0].split("\"")[1].split("<=")

            # if next node in dt_path is left child of current node, condition is '<='
            comparison = 'geq'
            if dt_path[i + 1] == children_left[node]:
                comparison = 'leq'

            feature = features[int(feature[2:-1])]

            if not feature in consultation_dict:
                consultation_dict[feature] = []
            consultation_dict[feature].append((comparison, thresh))

        # define symptom values consistent with thresholds derived from dt traversal
        consultation = []
        for q, v in consultation_dict.items():
            # if q is categorical we need to check if various thresholds were checked along the dt path
            if q in categorical_mapping:
                possible_vals = np.array(list(categorical_mapping[q].keys()))
                for ineq, thresh in v:
                    if ineq == "leq":
                        possible_vals = possible_vals[possible_vals <= float(thresh)]
                    elif ineq == "geq":
                        possible_vals = possible_vals[possible_vals >= float(thresh)]
                a = np.min(possible_vals)
            # if q is binary then we can directly deduce the correct value
            else:
                ineq, thresh = v[0]
                if ineq == "leq":
                    a = 0
                else:
                    a = 1
            consultation.append((q, a))

        # now check that symptom answers imposed by reverse traversal from disease leaf are at least x% correct wrt ground truth
        correct_ans = 0
        for q, a in consultation:
            if patient[q] == a:
                correct_ans += 1

        if correct_ans / len(consultation) >= min_correct_ans:
            found_valid_traversal = True
            break
        # else:
            # print("Traversal didn't yield enough valid symptom diagnoses, trying again")

    if found_valid_traversal:
        questions = [x[0] for x in consultation]
        answers = [x[1] for x in consultation]
        str_answers = [
            categorical_mapping[q][a] if q in categorical_mapping else ["False", "True"][a]
            for q, a in zip(questions, answers)
        ]

        return [(q, a) for q, a in zip(questions, str_answers)]

    else:
        print(f"Couldn't find a valid traversal ending at {targets[leaf_target]} with >= {min_correct_ans*100}% correct answers\n")
        print("Resorting to standard decision tree procedure for this patient")
        return decision_tree_consultation(clf, patient, features, categorical_mapping)


def plot_decision_tree(clf, data_dict, file_path):
    """
    https://scikit-learn.org/stable/modules/tree.html#classification
    """
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=data_dict["feature_names"],
        class_names=data_dict["target_names"],
        node_ids=True,
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
