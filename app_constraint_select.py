import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from scipy.spatial.distance import cosine, pdist, squareform
from dataset_utils import load_dataset

import random
import time
import pickle
import numpy as np

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
    "MNIST 2000 samples": "MNIST",
    "Country Indicators 1999": "COUNTRY-1999",
    "Country Indicators 2013": "COUNTRY-2013",
    "Country Indicators 2014": "COUNTRY-2014",
    "Country Indicators 2015": "COUNTRY-2015",
    "Breast Cancer Wisconsin (Diagnostic)": "BREAST-CANCER95",
    "Pima Indians Diabetes": "DIABETES",
    "Multidimensional Poverty Measures": "MPI"
}

image_datasets = ['MNIST-SMALL', 'MNIST', 'COIL20']

is_showing_image = False

# should start a separate static server for serving images:
# python -m http.server
static_host = 'http://0.0.0.0:8000'

app = dash.Dash()

app.css.append_css({
    "external_url": '{}/bootstrap.min.css'.format(static_host)
})

app.layout = html.Div([
    # dataset selection and debug info
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='datasetX',
                options=[{'label': k, 'value': v}
                         for k, v in datasets.items()],
                value=''
            ),
        ], className='col'),
        html.Div([
            html.Div(id='dataset-info', children='Dataset Info'),
        ], className='col')
    ], className='row mt-3'),

    # showing images or scatter plot
    html.Div(id='img-container', children=[], className='row mt-4'),

    html.Div(id='radar-container', children=[
        dcc.Graph(
            id='scatterX',
            config={
                'displayModeBar': False
            }
        ),
    ], className='row mt-4'),

    # control buttons
    html.Div([
        html.Button('Mustlink', id='btn-mustlink',
                    className="btn btn-outline-primary mx-auto"),
        html.Button('CannotLink', id='btn-cannotlink',
                    className="btn btn-outline-secondary mx-auto"),
        html.Button('Next', id='btn-next',
                    className="btn btn-outline-info mx-auto"),
        html.Button('Done', id='btn-done',
                    className="btn btn-outline-success mx-auto"),
        html.Button('Reset', id='btn-reset',
                    className="btn btn-outline-danger mx-auto"),
    ], className='row  mt-3'),

    # list of selected constraints
    html.Div([
        html.Div([html.Div(id='tbl-mustlinks')], className='col'),
        html.Div([html.Div(id='tbl-cannotlinks')], className='col')
    ], className='row mt-3'),

], className='container')


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

    _reset()

    global dataset_name
    global dataX
    global target_labels
    global target_names
    global dists

    dataset_name = name
    dataX, target_labels, target_names = load_dataset(dataset_name)
    dists = squareform(pdist(dataX))
    return 'N = {} instances, D = {} features'.format(*dataX.shape)


def _rand_pair(n_max):
    i1 = random.randint(0, n_max - 1)
    i2 = random.randint(0, n_max - 1)
    return (i1, i2) if i1 != i2 else _rand_pair(n_max)


@app.callback(dash.dependencies.Output('img-container', 'children'),
              [dash.dependencies.Input('btn-next', 'n_clicks')])
def show_pair_images(n_clicks):
    if dataX is None or target_names is None or not is_showing_image:
        return []

    n = dataX.shape[0]
    i1, i2 = _rand_pair(n)
    current_pair['id1'] = i1
    current_pair['id2'] = i2

    img_path = '{}/{}.svg'.format(static_host, dataset_name)
    return [
        html.Div([
            html.Img(src='{}#{}'.format(img_path, i1), height=100),
        ], className='col'),
        html.Div([
            html.Img(src='{}#{}'.format(img_path, i2), height=100),
        ], className='col')
    ]


@app.callback(dash.dependencies.Output('scatterX', 'figure'),
              [dash.dependencies.Input('btn-next', 'n_clicks')])
def show_pair_in_radar(n_clicks):
    if dataX is None or target_names is None or is_showing_image:
        return {'data': []}

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
                showticklabels=False,
                ticks=''
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", xanchor="center", x=0.5)
    )

    return {'data': data, 'layout': layout}


@app.callback(
    dash.dependencies.Output('tbl-mustlinks', 'children'),
    [dash.dependencies.Input('btn-mustlink', 'n_clicks')])
def select_ml(n_clicks):
    global mustlinks
    id1, id2 = current_pair['id1'], current_pair['id2']
    if id1 != -1 and id2 != -1:
        mustlinks.append([id1, id2])

    res = html.Table()
    if is_showing_image:
        res = _gen_img_table(mustlinks, is_mustlink=True)
    else:
        res = _gen_chart_table(mustlinks, is_mustlink=True)
    return res


@app.callback(
    dash.dependencies.Output('tbl-cannotlinks', 'children'),
    [dash.dependencies.Input('btn-cannotlink', 'n_clicks')])
def select_cl(n_clicks):
    global cannotlinks
    id1, id2 = current_pair['id1'], current_pair['id2']
    if id1 != -1 and id2 != -1:
        cannotlinks.append([id1, id2])

    res = html.Table()
    if is_showing_image:
        res = _gen_img_table(cannotlinks, is_mustlink=False)
    else:
        res = _gen_chart_table(cannotlinks, is_mustlink=False)
    return res


@app.callback(
    dash.dependencies.Output('btn-done', 'value'),
    [dash.dependencies.Input('btn-done', 'n_clicks')])
def save_links(_):
    if not dataset_name:
        return ''

    if mustlinks or cannotlinks:
        out_name = './output/manual_constraints/{}_{}.pkl'.format(
            dataset_name, time.strftime("%Y%m%d_%H%M%S"))
        data = {'mustlinks': mustlinks, 'cannotlinks': cannotlinks}
        pickle.dump(data, open(out_name, 'wb'))
        _reset()


@app.callback(
    dash.dependencies.Output('datasetX', 'value'),
    [dash.dependencies.Input('btn-reset', 'n_clicks')])
def reset_dataset(_):
    _reset()
    return ''


def _gen_img_table(links, is_mustlink):
    if len(links) == 0:
        return html.Table()

    img_path = '{}/{}.svg'.format(static_host, dataset_name)
    return html.Table(
        # Header
        [html.Tr([html.Th('Image 1'), html.Th('Image 2')])] +
        # Body
        [html.Tr([
            html.Td(html.Img(src='{}#{}'.format(img_path, i1), height=32)),
            html.Td(html.Img(src='{}#{}'.format(img_path, i2), height=32)),
        ]) for i1, i2 in links[::-1]],
        # bootstrap css
        style={'color': '#007bff' if is_mustlink else '#545b62'},
        className="table table-sm"
    )


def _gen_chart_table(links, is_mustlink):
    if len(links) == 0:
        return html.Table()

    text_color = '#007bff' if is_mustlink else '#545b62'
    layout = go.Layout(
        height=90,
        xaxis=dict(
            autorange=True, showgrid=False, zeroline=False, showline=False,
            autotick=True, ticks='', showticklabels=False
        ),
        yaxis=dict(
            autorange=True, showgrid=False, zeroline=False, showline=False,
            autotick=True, ticks='', showticklabels=False
        ),
        margin=go.Margin(l=0, r=0, b=0, t=0, pad=1),
        legend=dict(orientation="h", font=dict(color=text_color))
    )

    rows = []
    for i1, i2 in links[::-1]:
        d1, d2 = dataX[i1], dataX[i2]
        x_axis = np.arange(len(d1))
        trace1 = go.Scatter(x=x_axis, y=d1, line=dict(width=1),
                            name=target_names[i1][:20])
        trace2 = go.Scatter(x=x_axis, y=d2, line=dict(width=1),
                            name=target_names[i2][:20])

        chart = dcc.Graph(
            id='links-chart_{}_{}'.format(i1, i2),
            figure=go.Figure(data=[trace1, trace2], layout=layout),
            config={'displayModeBar': False}
        )
        rows.append(html.Tr([html.Td(chart)]))

    return html.Table(rows, className="table")


if __name__ == '__main__':
    app.run_server(debug=True)
