import argparse
import os
from ast import literal_eval

import matplotlib.pyplot as plt
import networkx as nx
import pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer


def encode_and_train(df: pandas.DataFrame) -> (MultiLabelBinarizer, GaussianNB):
    enc = MultiLabelBinarizer()
    print("Encoding data...")
    print(pandas.DataFrame(enc.fit_transform(df["present"] + df["member"].apply(str).apply(lambda x: [x])),
                           columns=enc.classes_, index=df.index))
    clf = GaussianNB()
    print("Training classifier...")
    clf.fit(enc.fit_transform(df["present"]), list(df["member"].apply(str)))
    print("Done.")
    return enc, clf


def graph_data(encoder: MultiLabelBinarizer, classifier: GaussianNB, noise_floor: float = 0):
    print("Building graph...")
    social_graph = nx.DiGraph()
    social_graph.add_nodes_from(encoder.classes_)
    for u in encoder.classes_:
        others = list(encoder.classes_)
        others.remove(u)
        # Create outgoing edges
        for o in others:
            vec = encoder.transform([[o]])
            prob_map = {encoder.classes_[n]: classifier.predict_proba(vec)[0][n] for n in
                        range(len(encoder.classes_))}
            social_graph.add_edge(u, o, weight=float(prob_map[u]))

    plt.subplot(121)
    mapping = {k: v for (k,v) in get_dict_from_namefile().items() if k in social_graph.nodes}
    nx.relabel_nodes(social_graph, mapping, copy=False)
    pos = nx.circular_layout(social_graph)
    edges, weights = zip(*[i for i in nx.get_edge_attributes(social_graph, 'weight').items() if i[1] > noise_floor])
    nx.draw(social_graph, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.get_cmap("viridis"), with_labels=True,
            arrowstyle='fancy')
    print("Done. Showing graph.")
    plt.show()
    print(sorted(social_graph.in_degree(weight='weight'), key=lambda x: x[1], reverse=True))


def get_dict_from_namefile():
    df = pandas.read_csv('names.csv', usecols=["member", "username"], index_col="member")
    namemap = {}
    for uid in df.index:
        namemap[str(uid)] = df.loc[uid]['username']
    return namemap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of file in the current working directory that contains the dataframe "
                                         "info")
    parser.add_argument("-nf", "--noise_floor", type=float)
    args = parser.parse_args()
    path = os.getcwd() + "\\" + args.filename
    print(f"Reading {path}...")
    df = pandas.read_csv(path)
    # pandas doesn't like saving lists; we have to rebuild `present` from string
    df['present'] = df['present'].apply(literal_eval)
    enc, clf = encode_and_train(df)

    if args.noise_floor:
        graph_data(enc, clf, args.noise_floor)
    else:
        graph_data(enc, clf)
