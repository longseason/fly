import os
from dotenv import load_dotenv
from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

load_dotenv()

def connection(api_token):
    c = Client('neuprint.janelia.org', dataset='male-cns:v1.0', token=api_token)
    return c

def grab_neurons():
    visual, _ = fetch_neurons(NC(type=['LC4', 'LPLC2'], status='Traced'))
    descending, _ = fetch_neurons(NC(type=['DNa02', 'DNp01'], status='Traced'))
    return visual, descending

def grab_edges(visuals, descending):
    neuron_df, conn_df = fetch_adjacencies(
        sources = visuals['bodyId'],
        targets = descending['bodyId'],
        min_total_weight = 0.5
    )
    return conn_df

def side_weights(conn_df, visual):
    grouped = conn_df.groupby(['bodyId_pre', 'bodyId_post'])
    pair_weights = grouped['weight'].sum()
    pair_weights = pair_weights.reset_index()

    sides = visual[['bodyId', 'instance']]
    merged = pair_weights.merge(sides, left_on='bodyId_pre', right_on='bodyId')

    merged['side'] = merged['instance'].str[-1]

    by_side = merged.groupby(['side','bodyId_post'])
    totals = by_side['weight'].sum()
    return totals

if __name__ == "__main__":
    client = connection(os.getenv('NEUPRINT_API_TOKEN'))
    visual, descending = grab_neurons()
    conn_df = grab_edges(visual, descending)
    print(len(visual), len(descending), len(conn_df))
    print(side_weights(conn_df, visual))