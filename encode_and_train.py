import argparse
import os
from ast import literal_eval

import matplotlib.pyplot as plt
import networkx as nx
import pandas
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


def plot_roc_auc(fpr, tpr, roc_auc):
    plt.figure()
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


def compute_roc_auc(n_classes, y_test, y_score):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def encode_and_train(df: pandas.DataFrame) -> (MultiLabelBinarizer, GaussianNB):
    mlb = MultiLabelBinarizer()
    print("Encoding data...")
    print(pandas.DataFrame(mlb.fit_transform(df["present"] + df["member"].apply(str).apply(lambda x: [x])),
                           columns=mlb.classes_, index=df.index))
    enc = LabelEncoder()
    X, y = mlb.transform(df["present"]), enc.fit_transform(df["member"].apply(str))
    print("Training svm...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    svc = svm.SVC(C=1.1, kernel="linear", probability=True, class_weight='balanced')
    svc.fit(X_train, y_train)
    print(
        f"Cross-validation SVC : {cross_val_score(svc, X_test, y_test, cv=KFold(n_splits=5), n_jobs=-1)}")  # Note: SVC lookin' better
    print("Done.")
    return mlb, svc, (X, y, X_train, X_test, y_train, y_test), enc


def graph_data(binarizer: MultiLabelBinarizer, encoder: LabelEncoder, classifier: GaussianNB, noise_floor: float = 0, name_file=None):
    print("Building graph...")
    social_graph = nx.DiGraph()
    social_graph.add_nodes_from(binarizer.classes_)
    for u in classifier.classes_:
        u = encoder.inverse_transform(u)
        others = list(binarizer.classes_)
        others.remove(u)
        # Create outgoing edges
        for o in others:
            vec = binarizer.transform([[o]])
            if encoder.transform([o]) in classifier.classes_:
                prob_map = {encoder.inverse_transform(classifier.classes_[n]): classifier.predict_proba(vec)[0][n] for
                            n in range(len(classifier.classes_))}
                weight = float(prob_map[u])
            else:
                weight = 0
            if weight > noise_floor:
                social_graph.add_edge(u, o, weight=weight)

    plt.subplot(121)
    if name_file:
        mapping = {k: v for (k, v) in get_dict_from_namefile(name_file).items() if k in social_graph.nodes}
        nx.relabel_nodes(social_graph, mapping, copy=False)
    print("In-degree weight sums:")
    print(sorted(social_graph.in_degree(weight='weight'), key=lambda x: x[1], reverse=True))
    # pos = nx.circular_layout(social_graph)
    pos = nx.fruchterman_reingold_layout(social_graph)
    edges, weights = zip(*[i for i in nx.get_edge_attributes(social_graph, 'weight').items() if i[1] > noise_floor])
    nx.draw(social_graph, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.get_cmap("winter"), with_labels=True,
            arrowstyle='fancy')
    print("Done. Showing graph.")
    plt.show()
    return social_graph


def save_as_graphml(graph, filename):
    print("Saving graph...")
    for node in graph.nodes():
        graph.node[node]['label'] = node
    nx.write_graphml(graph, filename)
    print("Done.")


def get_dict_from_namefile(file):
    df = pandas.read_csv(file, usecols=["member", "username"], index_col="member")
    namemap = {}
    for uid in df.index:
        namemap[str(uid)] = df.loc[uid]['username']
    return namemap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of file in the current working directory that contains the dataframe "
                                         "info", type=str)
    parser.add_argument("-nf", "--noise_floor", type=float, help="Cull edges below a certain weight. Only affects "
                                                                 "plot view.")
    parser.add_argument("-n", "--names", help="Name of csv file mapping Discord IDs and Usernames")
    parser.add_argument("-s", "--save_file", type=str, help="Filename to save as .graphml")
    args = parser.parse_args()
    path = os.getcwd() + "\\" + args.filename
    print(f"Reading {path}...")
    df = pandas.read_csv(path)
    # pandas doesn't like saving lists; we have to rebuild `present` from string
    df['present'] = df['present'].apply(literal_eval)
    mlb, clf, split_data, enc = encode_and_train(df)
    X, y, X_train, X_test, y_train, y_test = split_data

    # Generate Social Graph
    if args.noise_floor:
        graph = graph_data(mlb, enc, clf, args.noise_floor, name_file=args.names)
    else:
        graph = graph_data(mlb, enc, clf, name_file=args.names)

    # Save File
    if args.save_file:
        save_as_graphml(graph, args.save_file)

    y_score = clf.decision_function(X_test)
    y_test = mlb.transform([[enc.inverse_transform(i)] for i in y_test])
    fpr, tpr, roc_auc = compute_roc_auc(len(clf.classes_), y_test, y_score)
    plot_roc_auc(fpr, tpr, roc_auc)