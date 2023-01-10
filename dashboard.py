import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("agg")
# We're gonna use the Figure constructor
# see https://matplotlib.org/stable/gallery/user_interfaces/web_application_server_sgskip.html
# for more information
from matplotlib.figure import Figure
import requests

st.set_page_config(layout="wide")


@st.cache
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

# Chargement des data
@st.cache(allow_output_mutation=True)
def load_X_y(nan):
    X = pd.read_csv('https://projet7-bucket.s3.eu-west-3.amazonaws.com/X.csv', index_col=0).fillna(nan)
    y = pd.read_csv('https://projet7-bucket.s3.eu-west-3.amazonaws.com/y.csv', index_col=0)['TARGET']
    return X, y

nan = 1.01010101 # remplacement des NaN par cette valeur
X, y = load_X_y(nan)

# First block
left_column, middle_column, right_column = st.columns(3)

with left_column:
    # Choix du client par son id
    client_id = st.number_input("Client ID", key="client_id", min_value=100001, step=1)

if client_id > 100001:
    data_json = {'data': X.loc[client_id].to_list(),
                 'features_name': list(X.columns)}

    response_json = request_prediction(model_uri, data_json)
    #pred, global_features, global_vals, local_features, local_vals, X_imp = response_json.values()
    X_imp, global_features, global_vals, local_features, local_vals, pred = response_json.values()
    X_imp = pd.DataFrame(X_imp, columns=X.columns)

    #       First part

    with left_column:

        # Client Status

        if y.loc[client_id] == 0:
            status = '<span style="color:green">Solvable.</span>'
        else:
            status = '<span style="color:red">Insolvable.</span>'

        st.write('### Statut du client : ' + status, unsafe_allow_html=True)

        '#'
        '#'
        # Top n important features
        n_features = st.selectbox('Nombre de variables à afficher',
                                  range(5, 21),
                                  index=5,
                                  key='n_features')

    with middle_column:

        # Colorbar

        colors = ['r', 'darkorange', 'gold', 'limegreen', 'green']
        # définition de la barre d'échelle:
        cmap = (mpl.colors.ListedColormap(colors).with_extremes(over='0.25', under='0.75'))

        # Generate the figure **without using pyplot**.
        fig = Figure()
        ax = fig.subplots()
        #pred = 0.19
        color = cmap(pred)
        pred = round(pred * 100, 2)

        ax.text(50, 20, f'{pred}%', fontsize=20, color=color, ha='center', va='center')
        rect1 = mpl.patches.Rectangle((0, -10),
                                      pred, 20,
                                      color=color,
                                      #facecolor='w',
                                      #edgecolor='k'
                                      )

        rect2 = mpl.patches.Rectangle((0, -10),
                                      100, 20,
                                      facecolor='none',
                                      edgecolor='k'
                                      )

        ax.add_patch(rect1)
        ax.add_patch(rect2)

        ax.set_xlim([-10, 110])
        ax.set_ylim([-30, 40])
        ax.axis('off')
        ax.set_title('Prédiction du modèle', fontsize=20, fontweight='bold')
        ax.text(50, -20, 'que le client soit solvable', fontsize=18, ha='center', va='center')

        st.pyplot(fig)

    with right_column:
        '# \n' * 5
        if pred > 60:
            result = '<span style="color:green">Accordé</span>'
        else:
            result = '<span style="color:red">Refusé</span>'

        st.write('### Crédit ' + result, unsafe_allow_html=True)


    #       Feature importance

    local_features, local_vals = local_features[:n_features], local_vals[:n_features]
    global_features, global_vals = global_features[:n_features], global_vals[:n_features]

    left_column_2, right_column_2 = st.columns(2)
    # Local importance
    with left_column_2:
        fig = Figure(figsize=(8,6))
        ax = fig.subplots()
        ax.barh(local_features[::-1], local_vals[::-1],
                 color=["red" if coef < 0 else "green" for coef in local_vals[::-1]])
        # ax.set_xticks(rotation=30, horizontalalignment='right')
        x1 = - abs(1.1 * local_vals[0])
        x2 = - x1
        ax.set_xlim(x1, x2)
        ax.set_xlabel('Contribution')
        ax.set_title('Importance locale', fontsize=20, va='bottom')

        st.pyplot(fig)

    # Global importance
    with right_column_2:
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()
        ax.barh(global_features[::-1], global_vals[::-1],
                 color=["red" if coef < 0 else "green" for coef in global_vals[::-1]])
        x1 = - abs(1.1 * global_vals[0])
        x2 = - x1
        ax.set_xlim(x1, x2)
        ax.set_xlabel('Contribution')
        ax.set_title('Importance globale', fontsize=20, va='bottom')

        st.pyplot(fig)


    #   Distribution
    x = X.replace(nan, np.nan)

    left_column_3, right_column_3 = st.columns(2)
    with left_column_3:
        feature = st.selectbox('Sélectionnez une variable',
                               X.columns.sort_values(),
                               key='select_features')
        xscale = st.selectbox('Choisissez l\'échelle des abscisses',
                             ['Linéaire', 'Logarithmique'],
                             key='xscale_hist')
        yscale = st.selectbox('Choisissez l\'échelle des ordonnées',
                              ['Linéaire', 'Logarithmique'],
                              key='yscale_hist')
        clients = st.selectbox('Comparaison avec',
                               ['Tous les clients', 'Clients solvables', 'Clients insolvables'],
                               key='comparison')
        normalize = st.checkbox('Normaliser', value=True)

        if clients == 'Tous les clients':
            pass
        elif clients == 'Clients solvables':
            x = x[y == 0]
        else:
            x = x[y == 1]

    with right_column_3:
        if x[feature].unique().size < 10:
            n_bins = 10
        if (x[feature].unique().size >= 10) & (x[feature].unique().size < 100):
            n_bins = x[feature].unique().size
        if (x[feature].unique().size >= 100) & (x[feature].unique().size < 1000):
            n_bins = 100
        elif x[feature].unique().size > 1000:
            n_bins = x[feature].unique().size // 10
        n_bins = min(n_bins, 300)

        fig = Figure()
        ax = fig.subplots()
        if clients == 'Tous les clients':
            ax.hist(x.loc[y == 0, feature], bins=n_bins, density=normalize, label='Clients solvables')
            ax.hist(x.loc[y == 1, feature], bins=n_bins, density=normalize, label='Clients insolvables',
                    alpha=0.7)
        else:
            ax.hist(x[feature], bins=n_bins, density=normalize)

        if xscale == 'Logarithmique':
            ax.set_xscale('log')
        if yscale == 'Logarithmique':
            ax.set_yscale('log')

        xlims = ax.get_xlim()
        ax.plot(2 * [X_imp[feature]], ax.get_ylim(), 'r', alpha=0.5, linewidth=2,
                label='client')

        ax.set_title('Distribution des valeurs', fontsize=16)
        ax.set_xlabel(feature)
        ax.legend()
        ax.set_xlim(xlims)

        st.pyplot(fig)
