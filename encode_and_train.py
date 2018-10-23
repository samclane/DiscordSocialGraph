import argparse
import os
from ast import literal_eval

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


def preprocess(df: pandas.DataFrame):
    # Evaluate strings as lists
    df['present'] = df['present'].apply(literal_eval)
    # Remove members that only appear once
    df = df[df.groupby('member').member.transform(len) > 1].reset_index()
    return df


def svm_param_selection(X, y, nfolds):
    Cs = np.linspace(0.001, 10, 10)
    gammas = np.linspace(0.01, 1, 10)
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


def encode_and_train(df: pandas.DataFrame):
    # Encode "present" users as OneHotVectors
    mlb = MultiLabelBinarizer()
    print("Encoding data...")
    mlb.fit(df["present"] + df["member"].apply(str).apply(lambda x: [x]))

    # Encode user labels as ints
    enc = LabelEncoder()
    flat_member_list = df["member"].apply(str).append(pandas.Series(np.concatenate(df["present"]).ravel()))
    enc.fit(flat_member_list)
    X, y = mlb.transform(df["present"]), enc.transform(df["member"].apply(str))
    print("Training svm...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0, stratify=y)
    # params = svm_param_selection(X, y, 2)
    params = {'C': 10.0, 'gamma': 0.01}
    print(params)
    svc = svm.SVC(C=params['C'], gamma=params['gamma'], kernel="linear", probability=True)
    # svc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1)
    svc.fit(X_train, y_train)
    print(
        f"Cross-validation SVC : {cross_val_score(svc, X_test, y_test, cv=KFold(n_splits=5), n_jobs=-1)}")
    print("Done.")
    return mlb, svc, (X, y, X_train, X_test, y_train, y_test), enc, flat_member_list


def get_dict_from_namefile(file):
    df = pandas.read_csv(file, usecols=["member", "username"], index_col="member")
    namemap = {}
    for uid in df.index:
        namemap[str(uid)] = df.loc[uid]['username']
    return namemap


def graph_data(binarizer: MultiLabelBinarizer, encoder: LabelEncoder, classifier, member_list, noise_floor: float = 0,
               name_file=None):
    print("Building graph...")
    social_graph = nx.DiGraph()
    social_graph.add_nodes_from(encoder.classes_)
    for u in classifier.classes_:
        u = encoder.inverse_transform([u])[0]
        others = list(binarizer.classes_)
        others.remove(u)
        # Create outgoing edges
        for o in others:
            vec = binarizer.transform([[o]])
            if encoder.transform([o]) in classifier.classes_:
                prob_map = {encoder.inverse_transform([classifier.classes_[n]])[0]: classifier.predict_proba(vec)[0][n] for
                            n in range(len(classifier.classes_))}
                weight = float(prob_map[u]) * (1 + member_list.value_counts(normalize=True)[o])
            else:
                weight = 0
            social_graph.add_edge(u, o, weight=weight)
    # Prune useless nodes
    for n in list(social_graph.nodes):
        if social_graph.in_degree(weight='weight')[n] == 0 and social_graph.out_degree(weight='weight')[n] == 0:
            social_graph.remove_node(n)

    plt.subplot(121)
    if name_file:
        mapping = {k: v for (k, v) in get_dict_from_namefile(name_file).items() if k in social_graph.nodes}
        nx.relabel_nodes(social_graph, mapping, copy=False)
    print("In-degree weight sums:")
    print(sorted(social_graph.in_degree(weight='weight'), key=lambda x: x[1], reverse=True))
    pos = nx.circular_layout(social_graph)
    # pos = nx.fruchterman_reingold_layout(social_graph)
    edges, weights = zip(*[i for i in nx.get_edge_attributes(social_graph, 'weight').items() if i[1] > noise_floor])
    nx.draw(social_graph, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.get_cmap("winter"), with_labels=True,
            arrowstyle='fancy')
    print("Done. Showing graph.")
    return social_graph


def compute_roc_auc(n_classes, y_test, y_score):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    min_len = min(len(y_test.ravel()), len(y_score.ravel()))
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel()[:min_len], y_score.ravel()[:min_len])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def plot_roc_auc(fpr, tpr, roc_auc):
    plt.subplot(122)
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def save_as_graphml(graph, filename):
    print("Saving graph...")
    for node in graph.nodes():
        graph.node[node]['label'] = node
    nx.write_graphml(graph, filename)
    print("Done.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of file in the current working directory that contains the dataframe "
                                         "info", type=str)
    parser.add_argument("-nf", "--noise_floor", type=float, help="Cull edges below a certain weight. Only affects "
                                                                 "plot view.")
    parser.add_argument("-n", "--names", help="Name of csv file mapping Discord IDs and Usernames")
    parser.add_argument("-s", "--save_file", type=str, help="Filename to save as .graphml")
    return parser.parse_args()


def prep_graphs(data_file, noise_floor=0, name_file=None, save_file=None):
    path = os.getcwd() + "\\" + data_file
    print(f"Reading {path}...")
    df = pandas.read_csv(path)
    # pandas doesn't like saving lists; we have to rebuild `present` from string
    df = preprocess(df)
    mlb, clf, split_data, enc, member_list = encode_and_train(df)
    X, y, X_train, X_test, y_train, y_test = split_data

    # Generate Social Graph
    graph = graph_data(mlb, enc, clf, member_list, noise_floor, name_file=name_file)

    # Save File
    if save_file:
        save_as_graphml(graph, save_file)

    y_score = clf.decision_function(X_test)
    y_test = mlb.transform([[enc.inverse_transform([i])[0]] for i in y_test])
    fpr, tpr, roc_auc = compute_roc_auc(len(clf.classes_), y_test, y_score)
    plot_roc_auc(fpr, tpr, roc_auc)


if __name__ == "__main__":
    args = get_args()
    prep_graphs(args.filename, args.noise_floor, args.names, args.save_file)
