import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



executives = ["jeff.skilling@enron.com", "kenneth.lay@enron.com", "andrew.fastow@enron.com","rebecca.mark@enron.com","arthur.andersen@enron.com","lou.pai@enron.com"]

def get_executives():
    return executives


def create_network(df, title="", labels=False, no_plot=True,  executives = executives):
    G = nx.DiGraph()
    grouped_df = df.groupby(['sender', 'recipient1']).size().reset_index(name='count')
    for index, row in grouped_df.iterrows():
        if row['sender'] in executives or row['recipient1'] in executives:
            G.add_edge(row['sender'], row['recipient1'], weight=row['count'])
    if no_plot:
        return G
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.2, seed=42)
    nx.draw_networkx(G, pos, with_labels=labels,node_size=10, font_size=10, node_color='red')
    plt.title(title)
    plt.show()
    return G


def degree_centrality(G, executives=executives):
    cent = nx.degree_centrality(G)

    cent = nx.degree_centrality(G)
    name = []
    centrality = []

    for key, value in cent.items():
        name.append(key)
        centrality.append(value)

    cent = pd.DataFrame()    
    cent['name'] = name
    cent['centrality'] = centrality
    cent = cent.sort_values(by='centrality', ascending=False)
    cent = cent[~cent['name'].isin(executives)]

    plt.figure(figsize=(10, 12))
    _ = sns.barplot(x='centrality', y='name', data=cent[:5], orient='h')
    _ = plt.xlabel('Degree Centrality')
    _ = plt.ylabel('Correspondent')
    _ = plt.title('Top 5 Degree Centrality Scores in Enron Email Network')
    plt.show()
    return cent