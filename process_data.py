import os
import sys
from ast import literal_eval

import pandas
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer


def process_data(df: pandas.DataFrame):
    enc = MultiLabelBinarizer()
    print("Encoding data...")
    print(pandas.DataFrame(enc.fit_transform(df["present"]), columns=enc.classes_, index=df.index))
    clf = GaussianNB()
    print("Training classifier...")
    clf.fit(enc.fit_transform(df["present"]), list(df["member"]))
    return enc, clf


def graph_data(encoder: MultiLabelBinarizer, classifier: GaussianNB):
    social_graph = nx.DiGraph()
    social_graph.add_nodes_from(classifier.classes_)
    for u in classifier.classes_:
        others = list(classifier.classes_)
        others.remove(u)
        for o in others:
            vec = encoder.transform([[o]])
            prob_map = {classifier.classes_[n]: classifier.predict_proba(vec)[0][n] for n in range(len(classifier.classes_))}
            social_graph.add_edge(u, o, weight=float(prob_map[u]))

    plt.subplot(121)
    nx.draw(social_graph, with_labels=True, arrows=False, font_weight='bold')
    plt.show()

if __name__ == "__main__":
    df = pandas.read_csv(os.getcwd() + "\\" + str(sys.argv[1]))
    df['present'] = df['present'].apply(literal_eval)
    enc, clf = process_data(df)
    graph_data(enc, clf)