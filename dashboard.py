from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import matplotlib as mpl
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
fontsize = '24px'


########## Import des données ##########

def load_X_y(nan):
    X = pd.read_csv('/Users/gabriel/Documents/gabriel/Git/Projet-7/data/X.csv', index_col=0).fillna(nan)
    y = pd.read_csv('/Users/gabriel/Documents/gabriel/Git/Projet-7/data/y.csv', index_col=0)['TARGET']
    return X, y


nan = 1.01010101 # remplacement des NaN par cette valeur
X, y = load_X_y(nan)

index_values = X.index.tolist()


########## Appel au modèle ##########

def request_prediction(model_uri, data_json):
    headers = {"Content-Type": "application/json"}
    response = requests.request(method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

# Adresse du modèle
#model_uri = 'http://127.0.0.1:80/' # local
model_uri = 'http://35.180.69.239:80/' # cloud


###################
# PREMIERE PARTIE #
###################

########## Choix du client et status ##########
left_col1 = [daq.NumericInput(
    id='client-id',
    # value=0,
    min=100001,
    max=10000000,
    size=120,
),
    html.Br(),
    html.Span('Client ', style={'font-size': fontsize}),
    html.Span(id='client-status')]

########## Colorbar ##########
middle_col1 = [dcc.Graph(id='colorbar', style={'marginTop': '0em'})]

########## Status du crédit ##########
right_col1 = [html.Span('Crédit ', style={'font-size': fontsize}),
              html.Span(id='credit-status')]

########## Première ligne ##########
row1 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(left_col1, style={'textAlign': 'center'})),
                dbc.Col(html.Div(middle_col1), width=8),
                dbc.Col(html.Div(right_col1), style={'textAlign': 'left'}),
            ],
            align="center",
        ),
    ]
)

###################
# DEUXIEME PARTIE #
###################

row2 = html.Div(
    [
        dbc.Row(dbc.Col(html.Div('Nombre de variables', style={'padding': 20}))),
        dbc.Row(dbc.Col(dcc.Slider(5, 20, 1, value=10, id='n-features'), width=6)),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='local-importance')),
                dbc.Col(dcc.Graph(id='global-importance')),
            ]
        ),
    ]
)

###################
# TROISIEME PARTIE #
###################

########## Options du graphe ##########
options = html.Div(
    [
        html.H6('Variable', style={'margin-left': 10}),
        dcc.Dropdown(
            X.columns.sort_values(),
            'AMT_CREDIT',
            id='features'
        ),
        html.Br(),
        html.H6('Echelle des abscisses', style={'margin-left': 10}),
        dcc.Dropdown(
            ['Linéaire', 'Logarithmique'],
            'Linéaire',
            id='x-scale'
        ),
        html.Br(),
        html.H6('Echelle des ordonnées', style={'margin-left': 10}),
        dcc.Dropdown(
            ['Linéaire', 'Logarithmique'],
            'Linéaire',
            id='y-scale'
        ),
    ], style={'padding': 50}
)

row3 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(options),
                dbc.Col(dcc.Graph(id='feature-distribution'),
                        width=8),
            ]
        ),
    ]
)

###################
# APP LAYOUT #
###################

app.layout = html.Div(
    [
        dcc.Store(id='client-data'),
        row1,
        row2,
        row3,
    ]
)


@app.callback(
    Output('client-status', 'children'),
    Output('client-status', 'style'),
    Input('client-id', 'value')
)
def client_status(client_id):
    if client_id in index_values:
        if y.loc[client_id] == 0:
            status = 'Solvable'
            style = {'color': 'green', 'font-size': fontsize, 'font-weight': 'bold'}
        else:
            status = 'Insolvable'
            style = {'color': 'red', 'font-size': fontsize, 'font-weight': 'bold'}
    else:
        status = 'Inexistant'
        style = {'font-size': fontsize}
    return status, style


@app.callback(
    Output('client-data', 'data'),
    Input('client-id', 'value')
)
def model_prediction(client_id):
    if client_id in index_values:
        data_json = {'data': X.loc[client_id].to_list(),
                     'features_name': list(X.columns)}
        response_json = request_prediction(model_uri, data_json)
        return response_json


@app.callback(
    Output('colorbar', 'figure'),
    Input('client-id', 'value'),
    Input('client-data', 'data')
)
def make_colorbar(client_id, response_json):
    if client_id in index_values:
        colors = ['r', 'darkorange', 'gold', 'limegreen', 'green']
        # définition de la barre d'échelle:
        cmap = (mpl.colors.ListedColormap(colors).with_extremes(over='0.25', under='0.75'))
        pred = response_json['prediction']
        color = cmap(pred)
        rgba_tuple = tuple([0.99 if val == 1.0 else val for val in color])
        color = 'rgba' + str(rgba_tuple)
        pred_perc = round(pred * 100, 2)

        layout = go.Layout(template='simple_white', showlegend=False)
        colorbar = go.Figure(layout=layout)

        # Set axes properties
        colorbar.update_xaxes(range=[-20, 120], showgrid=False)
        colorbar.update_yaxes(range=[-5, 40])
        colorbar.update_xaxes(visible=False)
        colorbar.update_yaxes(visible=False)

        # Text
        colorbar.add_trace(go.Scatter(
            x=[50], y=[35],
            text=str(pred_perc) + '%',
            mode="text",
            textfont=dict(size=18, color=color)
        ))

        colorbar.add_trace(go.Scatter(
            x=[50], y=[5],
            text='Probabilité de remboursement',
            mode='text',
            textfont=dict(size=18)
        ))

        # filled rectangle
        colorbar.add_shape(type="rect",
                           x0=0, y0=10, x1=pred_perc, y1=30,
                           fillcolor=color,
                           opacity=1,
                           )

        # Unfilled rectangle for the edge
        colorbar.add_shape(type="rect",
                           x0=0, y0=10, x1=100, y1=30,
                           line=dict(
                               color="black",
                               width=2,
                           ),
                           fillcolor='rgba(0,0,0,0)',
                           opacity=1
                           )

        colorbar.update_xaxes(visible=False)
        colorbar.update_yaxes(visible=False)

        return colorbar

    else:
        layout = go.Layout(template='simple_white', showlegend=False)
        colorbar = go.Figure(layout=layout)

        # Set axes properties
        colorbar.update_xaxes(range=[-20, 120], showgrid=False)
        colorbar.update_yaxes(range=[-5, 40])

        colorbar.add_trace(go.Scatter(
            x=[50], y=[5],
            text='Probabilité de remboursement',
            mode='text',
            textfont=dict(size=18)
        ))

        # Unfilled rectangle for the edge
        colorbar.add_shape(type="rect",
                           x0=0, y0=10, x1=100, y1=30,
                           line=dict(
                               color="black",
                               width=2,
                           ),
                           fillcolor='rgba(0,0,0,0)',
                           opacity=1
                           )

        colorbar.update_xaxes(visible=False)
        colorbar.update_yaxes(visible=False)

        return colorbar


@app.callback(
    Output('credit-status', 'children'),
    Output('credit-status', 'style'),
    Input('client-id', 'value'),
    Input('client-data', 'data')
)
def client_status(client_id, response_json):
    if client_id in index_values:
        pred = response_json['prediction']
        if pred > 0.60:
            status = 'Accordé'
            style = {'color': 'green', 'font-size': fontsize, 'font-weight': 'bold'}
        else:
            status = 'Refusé'
            style = {'color': 'red', 'font-size': fontsize, 'font-weight': 'bold'}
    else:
        status = ' '
        style = {'font-size': fontsize}
    return status, style


@app.callback(
    Output('local-importance', 'figure'),
    Output('global-importance', 'figure'),
    Input('client-id', 'value'),
    Input('n-features', 'value'),
    Input('client-data', 'data')
)
def feature_importance(client_id, n_features, response_json):
    if client_id in index_values:
        local_values = response_json['local_imp_values'][:n_features]
        local_features = response_json['local_imp_features'][:n_features]
        global_values = response_json['global_imp_values'][:n_features]
        global_features = response_json['global_imp_features'][:n_features]

        layout = go.Layout(yaxis=dict(tickfont=dict(size=10)))

        local_importance = go.Figure(
            go.Bar(
                x=local_values[::-1],
                y=local_features[::-1],
                orientation='h',
                marker_color=['red' if coef < 0 else 'green' for coef in local_values[::-1]],
            ),
            layout=layout
        )

        global_importance = go.Figure(
            go.Bar(
                x=global_values[::-1],
                y=global_features[::-1],
                orientation='h',
                marker_color=['red' if coef < 0 else 'green' for coef in global_values[::-1]],
            ),
            layout=layout
        )

    else:
        local_importance = go.Figure()
        global_importance = go.Figure()

    return local_importance, global_importance


@app.callback(
    Output('feature-distribution', 'figure'),
    Input('client-id', 'value'),
    Input('client-data', 'data'),
    Input('features', 'value'),
    Input('x-scale', 'value'),
    Input('y-scale', 'value'),
)
def distribution(client_id, response_json, feature, x_scale, y_scale):
    if client_id in index_values:
        X_imp = pd.DataFrame(response_json['X_imputed'], columns=X.columns)

        log_x = False
        if x_scale == 'Logarithmique':
            log_x = True
        log_y = False
        if y_scale == 'Logarithmique':
            log_y = True

        if X[feature].unique().size < 10:
            n_bins = 10
        if (X[feature].unique().size >= 10) & (X[feature].unique().size < 100):
            n_bins = X[feature].unique().size
        if (X[feature].unique().size >= 100) & (X[feature].unique().size < 1000):
            n_bins = 100
        elif X[feature].unique().size > 1000:
            n_bins = X[feature].unique().size // 10
        n_bins = min(n_bins, 300)  # no more than 300 bins

        fig = px.histogram(x=X[feature], color=y, histnorm='percent', nbins=n_bins,
                           color_discrete_sequence=px.colors.qualitative.T10,
                           labels={
                               'x': feature,
                               'color': 'Status'
                           },
                           # template='seaborn',
                           log_x=log_x, log_y=log_y)
        fig.add_vline(X_imp[feature].values[0], line_width=2, line_color='red')

    else:
        fig = go.Figure()

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
