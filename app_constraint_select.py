import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from scipy.spatial.distance import cosine, pdist, squareform
from dataset_utils import load_dataset

import random
import time
import pickle


epsilon = 1e-5

dataX = None
target_labels = None
target_names = None
dists = None

dataset_name = ''
current_pair = {
    'id1': -1,
    'id2': -1
}

mustlinks = []
cannotlinks = []


def _reset():
    global dataX
    global target_labels
    global target_names
    global dists

    dataX = None
    target_labels = None
    target_names = None
    dists = None

    global dataset_name
    global current_pair

    dataset_name = ''
    current_pair = {
        'id1': -1,
        'id2': -1
    }

    global mustlinks
    global cannotlinks

    mustlinks = []
    cannotlinks = []


datasets = {
    "MNIST mini": "MNIST-SMALL",
    "COIL-20": "COIL20",
    "MNIST 2000 samples": "MNIST-2000",
    "Country Indicators 1999": "COUNTRY-1999",
    "Country Indicators 2013": "COUNTRY-2013",
    "Country Indicators 2014": "COUNTRY-2014",
    "Country Indicators 2015": "COUNTRY-2015",
    "Cars and Trucks 2004": "CARS04",
    "Breast Cancer Wisconsin (Diagnostic)": "BREAST-CANCER95",
    "Pima Indians Diabetes": "DIABETES",
    "Multidimensional Poverty Measures": "MPI"
}

image_datasets = ['MNIST-SMALL', 'COIL20']

is_showing_image = False

# should start a separate static server for serving images:
# python -m http.server
img_host = 'http://0.0.0.0:8000'

app = dash.Dash()
app.layout = html.Div([
    dcc.Dropdown(
        id='datasetX',
        options=[{'label': k, 'value': v} for k, v in datasets.items()],
        value=''
    ),
    html.Div(id='dataset-info', children='Dataset Info'),
    html.Div(id='ml-info', children='0 mustlink'),
    html.Div(id='cl-info', children='0 cannotlink'),
    html.Div(id='support-info', children='No output file'),

    html.Div(id='img-container', children=[
        html.Div(id='img-left', children='IMG LEFT'),
        html.Div(id='img-right', children='IMG RIGHT'),
    ], style={'columnCount': 2}, hidden=True),

    html.Div(id='radar-container', children=[
        dcc.Graph(
            id='scatterX'
        ),
    ], hidden=True),

    html.Button('Mustlink', id='btn-mustlink'),
    html.Button('CannotLink', id='btn-cannotlink'),
    html.Button('Next', id='btn-next'),
    html.Button('Done', id='btn-done'),
])


# ['children', 'id', 'n_clicks', 'key', 'accessKey', 'className',
#  'contentEditable', 'contextMenu', 'dir', 'draggable', 'hidden',
#  o'lang', 'spellCheck', 'style', 'tabIndex', 'title']

@app.callback(dash.dependencies.Output('img-container', 'hidden'),
              [dash.dependencies.Input('datasetX', 'value')])
def toggle_image_container(name):
    global is_showing_image
    global dataset_name
    is_showing_image = name in image_datasets
    dataset_name = name
    return not is_showing_image


@app.callback(dash.dependencies.Output('radar-container', 'hidden'),
              [dash.dependencies.Input('datasetX', 'value')])
def toggle_radar_container(name):
    global is_showing_image
    global dataset_name
    is_showing_image = name in image_datasets
    dataset_name = name
    return is_showing_image


@app.callback(dash.dependencies.Output('dataset-info', 'children'),
              [dash.dependencies.Input('datasetX', 'value')])
def update_dataset(name):
    if not name:
        return 'Please select a dataset!'

    global dataset_name
    global dataX
    global target_labels
    global target_names
    global dists

    dataset_name = name
    dataX, target_labels, target_names = load_dataset(dataset_name)
    # debug the number of classes
    print('Number of class: ', len(np.unique(target_labels)))
    # print(set(target_names))

    dists = squareform(pdist(dataX))
    dataset_info = """
        dataX: shape={}, mean={:.3f}, std={:.3f},
        min={:.3f}, max={:.3f},
        min_dist={:.3f}, max_dist={:.3f}
    """.format(dataX.shape, np.mean(dataX), np.std(dataX),
               np.min(dataX), np.max(dataX), np.min(dists), np.max(dists))
    return dataset_info


def _rand_pair(n_max):
    i1 = random.randint(0, n_max - 1)
    i2 = random.randint(0, n_max - 1)
    if i1 == i2:
        return _rand_pair(n_max)
    return (i1, i2)


@app.callback(dash.dependencies.Output('img-container', 'children'),
              [dash.dependencies.Input('btn-next', 'n_clicks')])
def show_pair_images(n_clicks):
    if dataX is None or target_names is None or not is_showing_image:
        return

    n = dataX.shape[0]
    i1, i2 = _rand_pair(n)
    current_pair['id1'] = i1
    current_pair['id2'] = i2

    img_path = '{}/{}.svg'.format(img_host, dataset_name)
    res = [
        html.Img(id='img-left', src='{}#{}'.format(img_path, i1)),
        html.Img(id='img-right', src='{}#{}'.format(img_path, i2)),
    ]
    return res


@app.callback(dash.dependencies.Output('scatterX', 'figure'),
              [dash.dependencies.Input('btn-next', 'n_clicks')])
def show_pair_in_radar(n_clicks):
    if dataX is None or target_names is None or is_showing_image:
        return

    n = dataX.shape[0]
    i1, i2 = _rand_pair(n)
    current_pair['id1'] = i1
    current_pair['id2'] = i2

    data1 = dataX[i1]
    data2 = dataX[i2]
    # cosine distance = 1 - cosine similarity
    sim1 = 1.0 - cosine(data1, data2)

    name1 = target_names[i1]
    name2 = target_names[i2]
    selected_idx = []
    for i in range(len(data1)):
        # consider the features that are different enough
        if abs(data1[i] - data2[i]) > epsilon:
            selected_idx.append(i)

    data1 = data1[selected_idx]
    data2 = data2[selected_idx]

    data1 = data1.tolist() + [data1[0]]
    data2 = data2.tolist() + [data2[0]]
    theta = ['f{}'.format(i) for i in range(len(data1))]

    max_val = max(max(data1), max(data2))

    data = [
        go.Scatterpolar(
            r=data1,
            theta=theta,
            fill='toself',
            name=name1,
        ),
        go.Scatterpolar(
            r=data2,
            theta=theta,
            fill='toself',
            name=name2
        ),
    ]

    layout = go.Layout(
        title="""
            {} distinguishable features <br>
            Cosine similarity = {:.4f}, Distance = {:.4f}
        """.format(len(data1) - 1, sim1, dists[i1, i2]),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.0, max_val]
            ),
            angularaxis=dict(
                visible=True,
                showticklabels=True
            )
        ),
        showlegend=True,
        legend=dict(orientation="h")
    )

    return {'data': data, 'layout': layout}


@app.callback(
    dash.dependencies.Output('ml-info', 'children'),
    [dash.dependencies.Input('btn-mustlink', 'n_clicks')])
def select_ml(n_clicks):
    global mustlinks
    id1, id2 = current_pair['id1'], current_pair['id2']
    if id1 != -1 and id2 != -1:
        mustlinks.append([id1, id2])
        # assert len(mustlinks) == n_clicks
        return '{} mustlinks'.format(n_clicks)


@app.callback(
    dash.dependencies.Output('cl-info', 'children'),
    [dash.dependencies.Input('btn-cannotlink', 'n_clicks')])
def select_cl(n_clicks):
    global cannotlinks
    id1, id2 = current_pair['id1'], current_pair['id2']
    if id1 != -1 and id2 != -1:
        cannotlinks.append([id1, id2])
        # assert len(cannotlinks) == n_clicks
        return '{} cannotlinks'.format(n_clicks)


@app.callback(
    dash.dependencies.Output('support-info', 'children'),
    [dash.dependencies.Input('btn-done', 'n_clicks')])
def save_links(_):
    if not dataset_name:
        return

    if mustlinks or cannotlinks:
        out_name = './output/manual_constraints/{}_{}.pkl'.format(
            dataset_name, time.strftime("%Y%m%d_%H%M%S"))
        data = {'mustlinks': mustlinks, 'cannotlinks': cannotlinks}
        pickle.dump(data, open(out_name, 'wb'))

        _reset()

        # Debug
        pkl_data = pickle.load(open(out_name, 'rb'))
        print(pkl_data)

        return "Write constraints to {}".format(out_name)


if __name__ == '__main__':
    app.run_server(debug=True)
