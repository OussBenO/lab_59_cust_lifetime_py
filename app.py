# LL PRO BONUS: R SHINY APPLICATION ----
# BUSINESS SCIENCE LEARNING LABS ----
# LAB 59: CUSTOMER LIFETIME VALUE | PYTHON DASH ----
# ----

# LIBRARIES

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc


import plotly.express as px

import pandas as pd
import pathlib

# APP SETUP
external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

PLOT_BACKGROUND = 'rgba(0,0,0,0)'
PLOT_FONT_COLOR = 'white'

LOGO = "https://www.business-science.io/img/business-science-logo.png"

# PATHS
BASE_PATH = pathlib.Path(__file__).parent.resolve()
ART_PATH = BASE_PATH.joinpath("artifacts").resolve()

# DATA
predictions_df = pd.read_pickle(ART_PATH.joinpath("predictions_df.pkl"))

df = predictions_df \
    .assign(
        spend_actual_vs_pred = lambda x: x['spend_90_total'] - x['pred_spend'] 
    )

# LAYOUT
navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Customer Spend Prediction", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plotly.com",
        ),
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        dbc.Collapse(
            id="navbar-collapse", navbar=True, is_open=False
        ),
    ],
    color="dark",
    dark=True,
)

app.layout = html.Div(
    children = [
        navbar, 
        dcc.Graph(id='graph-slider'),
        dcc.Slider(
            id    = 'spend-slider',
            value = df['spend_actual_vs_pred'].max(),
            max   = df['spend_actual_vs_pred'].max(),
            min   = df['spend_actual_vs_pred'].min()
        )
    ]
)

# CALLBACKS 
@app.callback(
    Output('graph-slider', 'figure'),
    Input('spend-slider', 'value'))
def update_figure(spend_delta_max):
    
    df_filtered = df[df['spend_actual_vs_pred'] <= spend_delta_max]

    fig = px.scatter(
        data_frame=df_filtered,
        x = 'frequency',
        y = 'pred_prob',
        color = 'spend_actual_vs_pred', 
        color_continuous_midpoint=0, 
        opacity=0.5, 
        color_continuous_scale='IceFire', 
        hover_name='customer_id',
        hover_data=['spend_90_total', 'pred_spend'],
    ) \
        .update_layout(
            {
                'plot_bgcolor': PLOT_BACKGROUND,
                'paper_bgcolor':PLOT_BACKGROUND,
                'font_color': PLOT_FONT_COLOR,
                'height':700
            }
        ) \
        .update_traces(
            marker = dict(size = 12)
        )
    
    return fig

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)