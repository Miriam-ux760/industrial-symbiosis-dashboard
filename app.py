# Industrial Symbiosis Dashboard (Dash-based)

import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import io
import base64
import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
from collections import Counter, defaultdict
import numpy as np

# --- Load and clean data ---
df = pd.read_excel("industrial_symbiosis_data.rEV.xlsx", sheet_name="industrial_symbiosis_data")
df = df.dropna(subset=['Company Name', 'Resource Needs', 'By-Products ', 'Latitude', 'Longitude'])
df['resource_list'] = df['Resource Needs'].str.lower().str.split('|').apply(lambda lst: [x.strip() for x in lst])
df['byproduct_list'] = df['By-Products '].str.lower().str.split('|').apply(lambda lst: [x.strip() for x in lst])

# --- Match resources ---
def find_matches(df, max_distance_km=100):
    matches = []
    for i, source in df.iterrows():
        for j, target in df.iterrows():
            if i == j:
                continue
            matched_items = set(source['byproduct_list']) & set(target['resource_list'])
            if matched_items:
                distance = geodesic((source['Latitude'], source['Longitude']), (target['Latitude'], target['Longitude'])).km
                if distance <= max_distance_km:
                    matches.append({
                        'From': source['Company Name'],
                        'To': target['Company Name'],
                        'Matched Items': ', '.join(matched_items),
                        'Distance (km)': round(distance, 2),
                        'From Industry': source.get('Industry', 'Unknown'),
                        'To Industry': target.get('Industry', 'Unknown')
                    })
    return matches

matches = find_matches(df)
matches_df = pd.DataFrame(matches)

# --- Setup Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

industries = sorted(df['Industry'].dropna().unique())
materials = sorted({m for lst in df['resource_list'].tolist() + df['byproduct_list'].tolist() for m in lst})

def compute_metrics(matches):
    if not matches:
        return 0, 0, 0, []
    total_matches = len(matches)
    companies = set()
    distances = []
    edge_count = Counter()

    for m in matches:
        companies.add(m['From'])
        companies.add(m['To'])
        distances.append(m['Distance (km)'])
        edge_count[m['From']] += 1
        edge_count[m['To']] += 1

    avg_distance = round(sum(distances) / len(distances), 2)
    top_companies = edge_count.most_common(5)
    return total_matches, len(companies), avg_distance, top_companies

def compute_mci(matches, df):
    matched_materials = set()
    for m in matches:
        matched_materials.update([mat.strip() for mat in m['Matched Items'].split(',')])
    all_byproducts = {mat for lst in df['byproduct_list'] for mat in lst}
    if not all_byproducts:
        return 0.0
    return round(len(matched_materials) / len(all_byproducts), 2)

def generate_flow_matrix(matches):
    matrix = defaultdict(lambda: defaultdict(int))
    material_map = defaultdict(lambda: defaultdict(set))
    for m in matches:
        matrix[m['From Industry']][m['To Industry']] += 1
        material_map[m['From Industry']][m['To Industry']].update(m['Matched Items'].split(', '))
    df_matrix = pd.DataFrame(matrix).fillna(0).astype(int).T
    df_materials = pd.DataFrame({k: {kk: ', '.join(vv) for kk, vv in v.items()} for k, v in material_map.items()}).T
    return df_matrix, df_materials

app.layout = html.Div([
    html.H1("Industrial Symbiosis Dashboard"),

    dbc.Row([
        dbc.Col(html.Div([
            html.Label("Filter by Industry Type:"),
            dcc.Dropdown(
                options=[{"label": i, "value": i} for i in industries],
                id="industry-filter",
                multi=True
            ),
            html.Label("Filter by Material:"),
            dcc.Dropdown(
                options=[{"label": m, "value": m} for m in materials],
                id="material-filter",
                multi=True
            ),
            html.Label("Distance (km):"),
            dcc.Slider(0, 150, 5, value=100, marks={i: str(i) for i in range(0, 151, 25)}, id="distance-slider")
        ]), width=3),

        dbc.Col(html.Div([
            html.Div(id="metrics-panel"),
            html.Div(id="mci-panel"),
            dcc.Graph(id="network-graph"),
            dcc.Graph(id="map-graph")
        ]), width=9)
    ]),

    html.Div([
        html.H3("Matching Table"),
        dash_table.DataTable(
            id="matches-table",
            columns=[{"name": i, "id": i} for i in matches_df.columns],
            data=matches_df.to_dict("records"),
            page_size=10,
            style_table={"overflowX": "auto"},
            filter_action="native",
            sort_action="native"
        )
    ], style={"padding": 20}),

    html.Div([
        html.H3("Industry-to-Industry Flow Matrix"),
        dcc.Graph(id="industry-heatmap"),
        dash_table.DataTable(id="industry-matrix-table", style_table={"overflowX": "auto"})
    ], style={"padding": 20})
])

@app.callback(
    [
        Output("network-graph", "figure"),
        Output("matches-table", "data"),
        Output("metrics-panel", "children"),
        Output("mci-panel", "children"),
        Output("industry-heatmap", "figure"),
        Output("industry-matrix-table", "data"),
        Output("industry-matrix-table", "columns"),
        Output("map-graph", "figure")
    ],
    [
        Input("industry-filter", "value"),
        Input("material-filter", "value"),
        Input("distance-slider", "value")
    ]
)
def update_dashboard(selected_industries, selected_materials, max_distance):
    filtered_df = df.copy()
    if selected_industries:
        filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]
    filtered_matches = find_matches(filtered_df, max_distance_km=max_distance)
    if selected_materials:
        filtered_matches = [m for m in filtered_matches if any(mat in m['Matched Items'] for mat in selected_materials)]

    total_matches, total_companies, avg_distance, top_companies = compute_metrics(filtered_matches)
    top_company_list = html.Ul([html.Li(f"{name}: {count} connections") for name, count in top_companies])
    metrics_panel = dbc.Card(dbc.CardBody([
        html.H5("Key Network Metrics", className="card-title"),
        html.P(f"Total Matches: {total_matches}"),
        html.P(f"Unique Companies Involved: {total_companies}"),
        html.P(f"Average Match Distance: {avg_distance} km"),
        html.P("Top 5 Most Connected Companies:"),
        top_company_list
    ]), style={"marginBottom": "20px"})

    mci_value = compute_mci(filtered_matches, filtered_df)
    mci_card = dbc.Card([
        dbc.CardBody([
            html.H5("Material Circularity Index (MCI)", className="card-title"),
            html.P(f"Current MCI: {mci_value}"),
            dcc.Graph(figure=go.Figure(go.Indicator(
                mode="gauge+number",
                value=mci_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "MCI (0 = Linear, 1 = Circular)"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "#ffcccc"},
                        {'range': [0.3, 0.7], 'color': "#fff5cc"},
                        {'range': [0.7, 1], 'color': "#ccffcc"}
                    ]
                }
            )).update_layout(height=200))
        ])
    ], style={"marginBottom": "20px"})

    # Network graph with arrow annotations
    G = nx.DiGraph()
    for match in filtered_matches:
        G.add_edge(match['From'], match['To'], label=match['Matched Items'], distance=match['Distance (km)'])
    pos = nx.spring_layout(G, seed=42)
    edge_trace = []
    annotations = []

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=1, color='gray'),
            hoverinfo='text',
            text=f"{u} → {v}<br>Items: {d['label']}<br>Distance: {d['distance']} km",
            mode='lines'
        ))
        # Add arrow annotation
        arrow_x = x0 + (x1 - x0) * 0.8
        arrow_y = y0 + (y1 - y0) * 0.8
        annotations.append(
            dict(
                ax=x0, ay=y0, axref='x', ayref='y',
                x=arrow_x, y=arrow_y, xref='x', yref='y',
                showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1, arrowcolor='black'
            )
        )

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        text=list(G.nodes()),
        mode='markers+text',
        textposition="top center",
        marker=dict(size=12, color='skyblue', line=dict(width=1, color='black'))
    )

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        title="Material Flow Network (with Direction Arrows)",
                        annotations=annotations
                    ))

    # Matrix and heatmap
    matrix_df, matrix_materials = generate_flow_matrix(filtered_matches)
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns,
        y=matrix_df.index,
        text=matrix_materials.values,
        hoverinfo="text",
        colorscale='Viridis'))
    heatmap_fig.update_layout(title="Industry-to-Industry Material Flow", xaxis_title="To Industry", yaxis_title="From Industry")
    matrix_table_data = matrix_df.reset_index().to_dict('records')
    matrix_table_columns = [{"name": i, "id": i} for i in matrix_df.reset_index().columns]

    # Map graph with arrow-like flow lines
    map_data = []
    for match in filtered_matches:
        from_row = df[df["Company Name"] == match["From"]].iloc[0]
        to_row = df[df["Company Name"] == match["To"]].iloc[0]

        # Add arrow direction using a line from source to target
        map_data.append(go.Scattermap(
            lat=[from_row["Latitude"], to_row["Latitude"]],
            lon=[from_row["Longitude"], to_row["Longitude"]],
            mode="lines+markers",
            line=dict(width=2, color="blue"),
            marker=dict(size=6, color="red"),
            hoverinfo='text',
            text=f"{match['From']} → {match['To']}<br>{match['Matched Items']}"
        ))

    map_fig = go.Figure(map_data)
        # Calculate map bounds based on company coordinates
    lats = df["Latitude"]
    lons = df["Longitude"]
    lat_center = lats.mean()
    lon_center = lons.mean()

    lat_margin = (lats.max() - lats.min()) * 0.1
    lon_margin = (lons.max() - lons.min()) * 0.1

    map_fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={"lat": lat_center, "lon": lon_center},
            zoom=6,
            bounds=dict(
                west=lons.min() - lon_margin,
                east=lons.max() + lon_margin,
                south=lats.min() - lat_margin,
                north=lats.max() + lat_margin
            )
        ),
        height=600,
        title="Geographic Material Flow Map (Zoomed to Region)",
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )


    return fig, filtered_matches, metrics_panel, mci_card, heatmap_fig, matrix_table_data, matrix_table_columns, map_fig

if __name__ == '__main__':
    app.run(debug=True)

    
