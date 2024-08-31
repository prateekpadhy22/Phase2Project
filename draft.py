import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import io
from PIL import Image

plt.switch_backend('Agg')

file_path = 'filtered_dataset_22_23.csv'
df = pd.read_csv(file_path)

# Sample the dataset
sampled_dataset = df.sample(frac=0.15, random_state=1)

# Initial data sample
initial_sample_size = 100
df_sample = sampled_dataset.sample(n=initial_sample_size, random_state=1)

# Preprocess data for PCA
# Select all numerical features
features = sampled_dataset.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
x = sampled_dataset[features].dropna()
y = sampled_dataset['Severity'][x.index]

# Standardize the features
x = StandardScaler().fit_transform(x)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Severity'] = y.values

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the server for Gunicorn

app.layout = html.Div([
    html.H1("Interactive Traffic Accident Dashboard"),

    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            dcc.Tabs([
                dcc.Tab(label='Dataset Details', children=[
                    html.Div([
                        html.Label("Select Plot Type:"),
                        dcc.RadioItems(
                            id='plot-type',
                            options=[
                                {'label': 'PCA Plot', 'value': 'pca'},
                                {'label': 'Correlation Matrix', 'value': 'correlation'},
                                {'label': 'Regression Plot', 'value': 'regression'}
                            ],
                            value='pca'
                        ),
                        dcc.Graph(id='pca-and-correlation-plot')
                    ])
                ]),
                dcc.Tab(label='Scatter Plot', children=[
                    html.Div([
                        html.Label("Select X-axis:"),
                        dcc.Dropdown(
                            id='x-axis-dropdown',
                            options=[{'label': col, 'value': col} for col in sampled_dataset.columns],
                            value='Temperature(F)'
                        ),
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.Label("Select Y-axis:"),
                        dcc.Dropdown(
                            id='y-axis-dropdown',
                            options=[{'label': col, 'value': col} for col in sampled_dataset.columns],
                            value='Severity'
                        ),
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Button('Load More Data', id='load-more-button', n_clicks=0),
                    dcc.Store(id='current-sample-size', data=initial_sample_size),
                    dcc.Graph(id='scatter-plot')
                ]),
                dcc.Tab(label='Bar Plot', children=[
                    html.Div([
                        html.Label("Select a Category:"),
                        dcc.Checklist(
                            id='category-checklist',
                            options=[{'label': col, 'value': col} for col in
                                     sampled_dataset.select_dtypes(include=['object']).columns],
                            value=['Weather_Condition']
                        ),
                    ], style={'width': '100%', 'display': 'inline-block'}),
                    dcc.Graph(id='bar-plot')
                ]),
                dcc.Tab(label='State Incidents Plot', children=[
                    dcc.Graph(id='state-incidents-plot')
                ]),
                dcc.Tab(label='Pair Plot', children=[
                    html.Div([
                        html.Label("Select Variables for Pair Plot:"),
                        dcc.Checklist(
                            id='pairplot-variables',
                            options=[{'label': col, 'value': col} for col in features],
                            value=[features[0], features[1]]  # Default selection
                        ),
                    ], style={'width': '100%', 'display': 'inline-block'}),
                    dcc.Graph(id='pair-plot')
                ]),
                dcc.Tab(label='Incident Map', children=[
                    dcc.Graph(id='incident-map')
                ]),
                dcc.Tab(label='Pie Chart', children=[
                    html.Div([
                        html.Label("Select a Category for Pie Chart:"),
                        dcc.Dropdown(
                            id='piechart-dropdown',
                            options=[{'label': col, 'value': col} for col in sampled_dataset.columns],
                            value='State'
                        ),
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    dcc.Graph(id='pie-chart')
                ]),
                dcc.Tab(label='Analyzing Crime per State', children=[
                    html.Div([
                        html.Label("Select the plot to analyze the crime per state"),
                        dcc.RadioItems(
                            id='crime-plot-type',
                            options=[
                                {'label': 'Area Plot', 'value': 'area'},
                                {'label': 'Strip Plot', 'value': 'strip'}
                            ],
                            value='area'
                        ),
                        dcc.Graph(id='crime-plot')
                    ]),
                    html.Div([
                        html.Label("Select Plot Type:"),
                        dcc.RadioItems(
                            id='rating-plot-type',
                            options=[
                                {'label': '3D Plot', 'value': '3d'},
                                {'label': 'Contour Plot', 'value': 'contour'},
                                {'label': 'Box Plot', 'value': 'box'}
                            ],
                            value='3d'
                        ),
                        dcc.Graph(id='rating-plot')
                    ])
                ])
            ])
        ]
    )
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('bar-plot', 'figure'),
    Output('state-incidents-plot', 'figure'),
    Output('pca-and-correlation-plot', 'figure'),
    Output('pair-plot', 'figure'),
    Output('incident-map', 'figure'),
    Output('pie-chart', 'figure'),
    Output('crime-plot', 'figure'),
    Output('rating-plot', 'figure'),
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value'),
    Input('load-more-button', 'n_clicks'),
    Input('category-checklist', 'value'),
    Input('pairplot-variables', 'value'),
    Input('piechart-dropdown', 'value'),
    Input('crime-plot-type', 'value'),
    Input('rating-plot-type', 'value'),
    Input('plot-type', 'value'),
    State('current-sample-size', 'data')
)
def update_plots(x_col, y_col, n_clicks, selected_categories, pairplot_vars, pie_category, crime_plot_type, rating_plot_type, plot_type, current_sample_size):
    sample_size = initial_sample_size + n_clicks * 100
    df_sample = sampled_dataset.sample(n=sample_size, random_state=1)

    scatter_fig = px.scatter(df_sample, x=x_col, y=y_col, title=f'Scatter plot of {x_col} vs {y_col}')

    if not selected_categories:
        bar_fig = {}
    else:
        df_grouped = sampled_dataset.groupby(selected_categories).size().reset_index(name='counts')
        bar_fig = px.bar(df_grouped, x=selected_categories[0], y='counts', title='Bar plot of selected categories')

    state_df = sampled_dataset.groupby('State').size().reset_index(name='counts')
    state_fig = px.choropleth(state_df, locations='State', locationmode='USA-states', color='counts',
                              scope='usa', title='State-wise Incidents')

    if plot_type == 'pca':
        pca_fig = px.scatter(pca_df, x='Principal Component 1', y='Principal Component 2', color='Severity',
                             title='PCA of All Features vs Severity')
        pca_fig.update_layout(
            title_font=dict(family='serif', color='blue', size=24),
            xaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
            yaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell")
        )
    elif plot_type == 'correlation':
        numeric_features = sampled_dataset.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
        correlation_matrix = numeric_features.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Features', fontsize=24, fontweight='medium', fontname='serif', color='blue')

        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        # Load the image from the buffer and convert it to a Plotly figure
        buf.seek(0)
        img = Image.open(buf)
        pca_fig = px.imshow(img)
        pca_fig.update_layout(
            title='Correlation Matrix of Numeric Features',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            margin=dict(l=20, r=20, t=40, b=20)
        )
    else:
        reg_plot = sns.lmplot(x='Start_Lat', y='Severity', data=df_sample, ci=None, height=8, aspect=1.5)
        reg_plot.set_axis_labels('Start_Lat', 'Severity')
        reg_plot.fig.suptitle('Regression Plot for Start_Lat vs Severity', fontsize=16, fontweight='bold', color='blue')

        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        reg_plot.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        pca_fig = px.imshow(img)
        pca_fig.update_layout(
            title='Regression Plot of Start_Lat vs Severity',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            margin=dict(l=20, r=20, t=40, b=20)
        )

    if len(pairplot_vars) < 2:
        pair_fig = px.scatter(title='Select at least two variables for pair plot')
    else:
        sns_pairplot = sns.pairplot(sampled_dataset[pairplot_vars])
        sns_pairplot.fig.suptitle('Pair Plot of Selected Variables', y=1.02, fontsize=16, fontweight='bold', color='blue')

        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        sns_pairplot.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        pair_fig = px.imshow(img)
        pair_fig.update_layout(
            title='Pair Plot of Selected Variables',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            margin=dict(l=20, r=20, t=40, b=20)
        )

    map_fig = px.scatter_mapbox(sampled_dataset, lat="Start_Lat", lon="Start_Lng", hover_name="Severity",
                                color="Severity", color_continuous_scale=[[0, 'yellow'], [1, 'red']], zoom=3, height=600)
    map_fig.update_layout(mapbox_style="open-street-map")
    map_fig.update_layout(
        title='Incident Map',
        title_font=dict(family='serif', color='blue', size=24),
        xaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
        yaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell"),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    pie_data = df_sample[pie_category].value_counts(normalize=True)
    pie_data = pie_data[pie_data > 0.01]
    other_data = 1 - pie_data.sum()
    if other_data > 0:
        pie_data['Other'] = other_data

    pie_fig = px.pie(values=pie_data.values, names=pie_data.index, title=f'Pie Chart of {pie_category}')
    pie_fig.update_layout(
        title_font=dict(family='serif', color='blue', size=24),
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell")
    )

    if crime_plot_type == 'area':
        crime_plot = px.area(df_sample, x='Start_Lat', y='Severity', title='Area Plot: Start_Lat vs Severity')
    else:  # plot_type == 'strip'
        crime_plot = px.strip(df_sample, x='Start_Lat', y='Severity', title='Strip Plot: Start_Lat vs Severity')

    crime_plot.update_layout(
        title_font=dict(family='serif', color='blue', size=24),
        xaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
        yaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell")
    )

    if rating_plot_type == '3d':
        rating_plot = px.scatter_3d(df_sample, x='Start_Lat', y='Severity', z='Start_Lng',
                                    title='3D Plot: Start_Lat, Severity, and Start_Lng')
    elif rating_plot_type == 'contour':
        rating_plot = go.Figure(go.Contour(
            z=df_sample['Severity'],
            x=df_sample['Start_Lat'],
            y=df_sample['Start_Lng'],
            colorscale='Viridis'
        ))
        rating_plot.update_layout(
            title='Contour Plot: Severity by Start_Lat and Start_Lng',
            title_font=dict(family='serif', color='blue', size=24),
            xaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
            yaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell")
        )
    else:  # rating_plot_type == 'box'
        rating_plot = px.box(df_sample, x='Severity', y='Start_Lat', title='Box Plot: Severity vs Start_Lat')

    rating_plot.update_layout(
        title_font=dict(family='serif', color='blue', size=24),
        xaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
        yaxis_title=dict(font=dict(family='serif', color='darkred', size=18)),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell")
    )

    return scatter_fig, bar_fig, state_fig, pca_fig, pair_fig, map_fig, pie_fig, crime_plot, rating_plot


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8080)
