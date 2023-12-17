
import dash as dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, State, Output
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import pprint
pp = pprint.PrettyPrinter(indent=4)
FA = "https://use.fontawesome.com/releases/v5.12.1/css/all.css"

path = 'Data_Cortex_Nuclear.xls'

# read in the data
df = pd.read_excel(path)
#df=pd.read_excel(path , engine='openpyxl')
na_columns = df.isna().sum() > 0

# fill in missing values
df.loc[:, na_columns.values] = df.loc[:, na_columns.values].fillna(df.loc[:, na_columns.values].mean())

# extract the requested sub-groups
cSCs = df[df.loc[:, 'class'] == 'c-SC-s']
tSCs = df[df.loc[:, 'class'] == 't-SC-s']

cSCs_features = cSCs.drop(columns=['MouseID', 'Genotype', 'Treatment', 'Behavior', 'class'])
tSCs_features = tSCs.drop(columns=['MouseID', 'Genotype', 'Treatment', 'Behavior', 'class'])
all_features = pd.concat((cSCs_features , tSCs_features))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_features)
pca_result = pd.DataFrame(pca_result, columns=['PCA 1', 'PCA 2'])
pca_result.loc[:135, 'group'] = 'c-SC-s'
pca_result.loc[135:, 'group'] = 't-SC-s'

iso = Isomap(n_neighbors=10, n_components=2)
iso_result = iso.fit_transform(all_features)
iso_result = pd.DataFrame(iso_result, columns=['ISOMAP 1', 'ISOMAP 2'])
iso_result.loc[:135, 'group'] = 'c-SC-s'
iso_result.loc[135:, 'group'] = 't-SC-s'

tsne = TSNE(n_components=2, perplexity=5)
tsne_result = tsne.fit_transform(all_features)
tsne_result = pd.DataFrame(tsne_result, columns=['tSNE 1', 'tSNE 2'])
tsne_result.loc[:135, 'group'] = 'c-SC-s'
tsne_result.loc[135:, 'group'] = 't-SC-s'
all_features = all_features.reset_index(drop=True)
# all_features = all_features.reset_index()


fig = px.scatter(pca_result, x='PCA 1', y='PCA 2', color='group')
fig_scatter = px.scatter(all_features, x=all_features.columns[0], y=all_features.columns[0],
                         color=['unselected']*len(all_features))
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP, FA]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#115511',
    'text': '#7FDBFF'
}
# creating the general layout
app.layout = html.Div(children=[
    html.H1(
        children='Interactive Analysis of Mice Protein Expressions',
        style={
            'textAlign': 'center',
            'paddingTop': '15px',  # Add padding to the top
            'paddingBottom': '15px',
            'color': 'phantom',
            'marginBottom': '0',
            'backgroundColor': 'turquoise',
        }
    ),

    html.Div(html.H4(children='Exploratory data analysis using interactive visualization Dashboard',
             style={
                 'textAlign': 'center',
                 'color': 'black',
                 'margin-bottom': '30px',
                 'margin-top': '0',
                 'paddingBottom': '15px',
                 'backgroundColor': 'turquoise',
             }
             )),
    html.Div([
        html.Div([
            dcc.RadioItems(
                id='reduction',
                options=[{'label': i, 'value': i} for i in ['PCA', 'ISOMAP', 't-SNE']],
                value='PCA',
                labelStyle={'display': 'inline-block', 'margin-left': '30px'}
            ),
        ],
            style={'width': '49%', 'textAlign': 'center'}),
        html.Div([

            html.Div([
                html.Div(children='x-Axis'),
                dcc.Dropdown(
                    id='x-axis',
                    options=[{'label': i, 'value': i} for i in all_features.columns],
                    value=all_features.columns[0],
                    style={'width': '200px', 'margin-top': '5px'}
                ),
            ]),
            html.Div([
                html.Div(children='y-Axis'),
                dcc.Dropdown(
                    id='y-axis',
                    options=[{'label': i, 'value': i} for i in all_features.columns],
                    value=all_features.columns[0],
                    style={'width': '200px', 'margin-top': '5px'}
                ),
            ]),
            html.Div([
                html.Div(children='132', style={'visibility': "hidden"}),
                html.Button("Add", id="add", style={'width': '200px', 'margin-top': '5px'})
            ])

        ],
            style={'width': '49%', 'textAlign': 'center', 'display': 'flex', 'justify-content': 'space-around'})
    ], style={'display': 'flex'}),

    html.Div([
        dcc.Graph(
            id='reduction-graph',
            style={
                'width': '49%'
            },
            figure=fig
        ),
        dcc.Graph(
            id={'type': 'Graphique', 'index': 0},
            style={
                'width': '49%'
            },
            figure=fig_scatter
        )

    ], style={'display': 'flex'}),
    html.Div(id="figure-container", style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'flex-start'},
             children=[]),
    dcc.Store(id="figure-col"),
])


def get_figure(x_col, y_col):
    fig = px.scatter(all_features, x=x_col, y=y_col)
    fig.update_layout(dragmode='select', hovermode=False)
    return fig.to_dict()


def create_container_el(x_axis, y_axis, i):
    return html.Div([
                html.Div(
                    dcc.Graph(
                        id={'type': 'Graphique',
                            'index': i + 1},
                        style={
                            'width': '65vh',
                        },
                        figure=get_figure(x_axis, y_axis)
                    )
                )
            ], style={'display': 'flex', 'flex-direction': 'column'})

# Updates the left plot
@app.callback(Output('reduction-graph', 'figure'),
              Input('reduction', 'value'))
def update_reduction(value):
    if value == 't-SNE':
        fig = px.scatter(tsne_result, x='tSNE 1', y='tSNE 2', color='group')
    elif value == 'ISOMAP':
        fig = px.scatter(iso_result, x='ISOMAP 1', y='ISOMAP 2', color='group')
    elif value == 'PCA':
        fig = px.scatter(pca_result, x='PCA 1', y='PCA 2', color='group')
    return fig

@app.callback([
    Output("figure-container", "children"),
    Output("figure-col", 'data'),
    Output({'type': 'Graphique', 'index': 0}, 'figure'),
    ],
    [
        Input("add", "n_clicks"),
        Input('x-axis', 'value'),
        Input('y-axis', 'value')
    ],
    [
        State("figure-col", 'data'),
    ], prevent_initial_call=True)
def edit_list(add, x_axis, y_axis, items):
    # Create color lists
    # Check if the add is fired
    triggered = [t["prop_id"] for t in dash.callback_context.triggered]
    adding = len([1 for i in triggered if i in "add.n_clicks"])
    if not items: items=[]
    if adding:
        items.append({'x_axis':x_axis, 'y_axis': y_axis})
    new_cont = []
    for i, el in enumerate(items):
        new_cont.append(create_container_el(el['x_axis'], el['y_axis'], i))
    return [new_cont, items, px.scatter(all_features, x_axis, y_axis)]





if __name__ == '__main__':
    app.run_server(debug=True, port=3010)
