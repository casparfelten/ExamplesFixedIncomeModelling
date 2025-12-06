#!/usr/bin/env python3
"""
QSIG Macro Graph Explorer - Professional Financial Interface
===========================================================

Bloomberg Terminal-style web interface for Bayesian graph models.

Features:
- Mode-based UI (Browse/Details/Perturbation)
- Floating side panel with live controls
- Real-time graph visualization with numbers
- Professional financial styling
- Live parameter perturbation with visual feedback

Usage:
    cd webapp
    source ../webapp_env/bin/activate
    python app.py
"""

import dash
from dash import dcc, html, Input, Output, State, callback, clientside_callback, ClientsideFunction
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
# Removed plotly imports - using pure Cytoscape only
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Direct file imports
import importlib.util

# Calculate paths carefully
current_file = Path(__file__).resolve()
webapp_dir = current_file.parent  
project_root = webapp_dir.parent

# Add the src directory to path so relative imports work
sys.path.insert(0, str(project_root / "src"))

# Now we can import normally
import graph.types as types_module
import graph.edge_runner as runner_module

# Import classes
Graph = types_module.Graph
EventContext = types_module.EventContext
NodeId = types_module.NodeId
NodeType = types_module.NodeType
EdgeRunner = runner_module.EdgeRunner

# Initialize Dash app with professional theme
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
], suppress_callback_exceptions=True)

# Load graph data
def load_graph() -> Graph:
    """Load the macro graph from registry.""" 
    graph_path = project_root / "registry" / "macro_graph.json"
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    return Graph.from_dict(graph_data)

def get_edge_runner() -> EdgeRunner:
    """Get edge runner for model execution."""
    graph = load_graph()
    return EdgeRunner(graph, cache_models=True)

# Global instances
graph = load_graph()
runner = get_edge_runner()

def create_graph_elements():
    """Create Cytoscape graph elements with background factors."""
    elements = []
    
    # Event and instrument nodes
    for node_name, node in graph.nodes.items():
        if node.type == NodeType.EVENT:
            elements.append({
                'data': {
                    'id': node_name,
                    'label': node_name,
                    'type': 'event',
                    'description': node.description
                },
                'classes': 'event-node'
            })
        else:
            elements.append({
                'data': {
                    'id': node_name, 
                    'label': node_name,
                    'type': 'instrument',
                    'description': node.description
                },
                'classes': 'instrument-node'
            })
    
    # Background factor nodes
    background_factors = set()
    for slot in graph.edge_slots.values():
        for feature in slot.feature_spec.get('background_features', []):
            background_factors.add(feature)
    
    for factor in background_factors:
        elements.append({
            'data': {
                'id': factor,
                'label': factor.replace('_', '\n'),
                'type': 'background_factor'
            },
            'classes': 'background-factor-node'
        })
    
    # Main edges (solid lines)
    for slot_id, slot in graph.edge_slots.items():
        elements.append({
            'data': {
                'id': slot_id,
                'source': slot.from_node.name,
                'target': slot.to_node.name,
                'label': slot_id,
                'type': 'main_edge',
                'description': slot.description,
                'edge_count': len(slot.edges)
            },
            'classes': 'main-edge'
        })
    
    # Background factor edges (dotted lines)
    for slot in graph.edge_slots.values():
        # Connect background factors to the from_node
        for feature in slot.feature_spec.get('background_features', []):
            elements.append({
                'data': {
                    'id': f"{feature}_{slot.from_node.name}",
                    'source': feature,
                    'target': slot.from_node.name,
                    'type': 'background_edge'
                },
                'classes': 'background-edge'
            })
    
    return elements

# Professional Financial Cytoscape stylesheet
cytoscape_stylesheet = [
    # Event nodes - Dark professional blue with shadows
    {
        'selector': '.event-node',
        'style': {
            'background-color': '#0f172a',
            'background-image': 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
            'border-color': '#3b82f6',
            'border-width': 4,
            'width': 120,
            'height': 120,
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'color': '#ffffff',
            'font-size': 16,
            'font-weight': 700,
            'font-family': 'Inter, sans-serif',
            'text-outline-color': '#000000',
            'text-outline-width': 2,
            'box-shadow': '0 8px 32px rgba(0, 0, 0, 0.3)'
        }
    },
    # Instrument nodes - Dark professional green
    {
        'selector': '.instrument-node',
        'style': {
            'background-color': '#022c22',
            'background-image': 'linear-gradient(135deg, #064e3b 0%, #022c22 100%)',
            'border-color': '#10b981',
            'border-width': 4,
            'width': 120,
            'height': 120,
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'color': '#ffffff',
            'font-size': 16,
            'font-weight': 700,
            'font-family': 'Inter, sans-serif',
            'text-outline-color': '#000000',
            'text-outline-width': 2,
            'box-shadow': '0 8px 32px rgba(0, 0, 0, 0.3)'
        }
    },
    # Background factor nodes - Muted with better visibility
    {
        'selector': '.background-factor-node',
        'style': {
            'background-color': '#334155',
            'border-color': '#64748b',
            'border-width': 2,
            'width': 80,
            'height': 80,
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'color': '#f1f5f9',
            'font-size': 11,
            'font-weight': 600,
            'font-family': 'Inter, sans-serif',
            'text-outline-color': '#1e293b',
            'text-outline-width': 1,
            'opacity': 0.9
        }
    },
    # Main edges - Thick financial gold
    {
        'selector': '.main-edge',
        'style': {
            'width': 6,
            'line-color': '#f59e0b',
            'target-arrow-color': '#f59e0b',
            'target-arrow-shape': 'triangle',
            'target-arrow-size': 20,
            'curve-style': 'straight',
            'label': 'data(label)',
            'font-size': 12,
            'font-weight': 700,
            'color': '#ffffff',
            'text-background-color': '#f59e0b',
            'text-background-opacity': 1,
            'text-background-padding': '4px',
            'text-border-color': '#d97706',
            'text-border-width': 1,
            'font-family': 'Inter, sans-serif'
        }
    },
    # Background edges - More prominent dotted
    {
        'selector': '.background-edge',
        'style': {
            'width': 3,
            'line-color': '#64748b',
            'line-style': 'dashed',
            'opacity': 0.7,
            'curve-style': 'straight'
        }
    },
    # Selected elements - Bright highlight
    {
        'selector': ':selected',
        'style': {
            'border-color': '#ef4444',
            'border-width': 6,
            'box-shadow': '0 0 20px rgba(239, 68, 68, 0.5)'
        }
    },
    # Faded nodes (perturbation mode)
    {
        'selector': '.faded',
        'style': {
            'opacity': 0.2
        }
    },
    # Active nodes with numbers - Bright highlighting
    {
        'selector': '.active-prediction',
        'style': {
            'border-color': '#ef4444',
            'border-width': 6,
            'box-shadow': '0 0 30px rgba(239, 68, 68, 0.7)',
            'font-size': 18,
            'font-weight': 700
        }
    }
]

# App layout
app.layout = dbc.Container([
    # Data stores
    dcc.Store(id='current-mode-store', data='default'),
    dcc.Store(id='selected-edge-store', data={}),
    dcc.Store(id='perturbation-data-store', data={}),
    dcc.Store(id='prediction-results-store', data={}),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("QSIG Macro Graph Explorer", 
                       style={'color': '#1e293b', 'fontWeight': '700', 'fontSize': '2.5rem', 'marginBottom': '0'}),
                html.P("Interactive Bayesian Graph Interface for Fixed Income Modeling", 
                       style={'color': '#64748b', 'fontSize': '1.1rem', 'fontWeight': '400', 'marginBottom': '0'}),
            ])
        ], width=10),
        dbc.Col([
            html.Div(id='mode-indicator', children=[
                dbc.Badge("Browse Mode", color="primary", pill=True, className="fs-6")
            ], style={'textAlign': 'right', 'marginTop': '20px'})
        ], width=2)
    ], className="mb-4", style={'padding': '20px 0', 'borderBottom': '2px solid #e2e8f0'}),
    
    # Main graph area with floating panel
    html.Div([
        # Graph visualization
        dbc.Card([
            dbc.CardBody([
                cyto.Cytoscape(
                    id='cytoscape-graph',
                    elements=create_graph_elements(),
                    stylesheet=cytoscape_stylesheet,
                    layout={
                        'name': 'preset',
                        'positions': {
                            'CPI': {'x': 100, 'y': 150},
                            'UNEMPLOYMENT': {'x': 100, 'y': 400},
                            'HY_OAS': {'x': 500, 'y': 100},
                            'VIX': {'x': 500, 'y': 250}, 
                            'YIELD_2Y': {'x': 500, 'y': 400},
                            'YIELD_10Y': {'x': 500, 'y': 550},
                            'fed_funds': {'x': 300, 'y': 50},
                            'slope_10y_2y': {'x': 300, 'y': 150},
                            'yield_vol_10y': {'x': 300, 'y': 250},
                            'yield_vol_2y': {'x': 300, 'y': 350},
                            'hy_vol': {'x': 300, 'y': 450},
                            'hy_oas_before': {'x': 300, 'y': 550},
                            'unemp_rate': {'x': 300, 'y': 650},
                            'vix_vol': {'x': 300, 'y': 750},
                            'vix_before': {'x': 300, 'y': 850},
                            'stlfsi': {'x': 300, 'y': 950}
                        },
                        'fit': True,
                        'padding': 50
                    },
                    style={'height': '700px', 'backgroundColor': '#f8fafc'},
                    responsive=True,
                    boxSelectionEnabled=False
                )
            ], style={'backgroundColor': '#ffffff', 'padding': '0'})
        ], style={'border': '1px solid #d1d5db', 'borderRadius': '12px', 'position': 'relative'}),
        
        # Floating side panel
        html.Div(id='floating-panel', children=[], style={
            'position': 'absolute',
            'top': '20px',
            'right': '20px',
            'width': '320px',
            'backgroundColor': 'rgba(255, 255, 255, 0.95)',
            'border': '1px solid #d1d5db',
            'borderRadius': '12px',
            'padding': '20px',
            'boxShadow': '0 10px 25px -5px rgba(0, 0, 0, 0.1)',
            'backdropFilter': 'blur(10px)',
            'zIndex': '1000',
            'display': 'none'
        })
    ], style={'position': 'relative', 'marginBottom': '20px'}),
    
    # Instructions overlay
    html.Div(id='instructions-overlay', children=[
        dbc.Alert([
            html.H6("🎯 How to Use", className="alert-heading"),
            html.P("• Click any edge to view model details", className="mb-1"),
            html.P("• Use perturbation mode to simulate scenarios", className="mb-1"),
            html.P("• Background factors show as dotted connections", className="mb-0")
        ], color="info", style={'margin': '0 auto', 'maxWidth': '600px'})
    ], style={'textAlign': 'center', 'marginTop': '20px'})
    
], fluid=True, style={
    'backgroundColor': '#f8fafc',
    'minHeight': '100vh',
    'fontFamily': 'Inter, sans-serif',
    'padding': '20px'
})

# Callback: Handle edge clicks and mode switching
@app.callback(
    [Output('current-mode-store', 'data'),
     Output('selected-edge-store', 'data'),
     Output('floating-panel', 'style'),
     Output('mode-indicator', 'children')],
    [Input('cytoscape-graph', 'selectedEdgeData')],
    [State('current-mode-store', 'data')]
)
def handle_edge_selection(selected_edges, current_mode):
    """Handle edge selection and mode switching."""
    if selected_edges and len(selected_edges) > 0:
        edge = selected_edges[0]
        if edge.get('type') == 'main_edge':
            # Switch to edge details mode
            panel_style = {
                'position': 'absolute',
                'top': '20px',
                'right': '20px',
                'width': '320px',
                'backgroundColor': 'rgba(255, 255, 255, 0.95)',
                'border': '1px solid #d1d5db',
                'borderRadius': '12px',
                'padding': '20px',
                'boxShadow': '0 10px 25px -5px rgba(0, 0, 0, 0.1)',
                'backdropFilter': 'blur(10px)',
                'zIndex': '1000',
                'display': 'block'
            }
            mode_badge = dbc.Badge("Edge Details", color="success", pill=True, className="fs-6")
            return 'edge-details', {'slot_id': edge['id'], 'label': edge['label']}, panel_style, mode_badge
    
    # Default state
    panel_style = {
        'position': 'absolute',
        'top': '20px',
        'right': '20px', 
        'width': '320px',
        'backgroundColor': 'rgba(255, 255, 255, 0.95)',
        'border': '1px solid #d1d5db',
        'borderRadius': '12px',
        'padding': '20px',
        'boxShadow': '0 10px 25px -5px rgba(0, 0, 0, 0.1)',
        'backdropFilter': 'blur(10px)',
        'zIndex': '1000',
        'display': 'none'
    }
    mode_badge = dbc.Badge("Browse Mode", color="primary", pill=True, className="fs-6")
    return 'default', {}, panel_style, mode_badge

# Callback: Update floating panel content
@app.callback(
    Output('floating-panel', 'children'),
    [Input('current-mode-store', 'data'),
     Input('selected-edge-store', 'data')]
)
def update_floating_panel(current_mode, selected_edge_data):
    """Update the floating panel content based on current mode."""
    if current_mode == 'edge-details' and 'slot_id' in selected_edge_data:
        slot_id = selected_edge_data['slot_id']
        
        try:
            slot = graph.edge_slots[slot_id]
            
            content = [
                # Header
                html.Div([
                    html.H5(slot_id, style={'color': '#1e293b', 'fontWeight': '600', 'marginBottom': '5px'}),
                    html.P(slot.description, style={'color': '#64748b', 'fontSize': '0.9rem', 'marginBottom': '15px'}),
                    dbc.Button("Enter Perturbation Mode", id='enter-perturbation-btn', 
                              color="warning", size="sm", className="w-100 mb-3")
                ]),
                
                # Edge details
                html.Div([
                    html.H6("Connection", style={'color': '#1e293b', 'fontWeight': '500', 'marginBottom': '10px'}),
                    html.P([html.Strong("From: "), f"{slot.from_node.name}"], 
                           style={'fontSize': '0.85rem', 'marginBottom': '5px'}),
                    html.P([html.Strong("To: "), f"{slot.to_node.name}"], 
                           style={'fontSize': '0.85rem', 'marginBottom': '15px'}),
                ]),
                
                # Features
                html.Div([
                    html.H6("Features", style={'color': '#1e293b', 'fontWeight': '500', 'marginBottom': '10px'}),
                    html.P([html.Strong("Active: "), html.Code(slot.feature_spec['active_factor'])],
                           style={'fontSize': '0.85rem', 'marginBottom': '8px'}),
                    html.P("Background:", style={'fontSize': '0.85rem', 'fontWeight': '500', 'marginBottom': '5px'}),
                    html.Ul([
                        html.Li(html.Code(f, style={'fontSize': '0.8rem'})) 
                        for f in slot.feature_spec['background_features']
                    ], style={'fontSize': '0.8rem', 'marginBottom': '15px'})
                ]),
                
                # Model performance
                html.Div([
                    html.H6("Models", style={'color': '#1e293b', 'fontWeight': '500', 'marginBottom': '10px'}),
                    *[create_model_summary(edge) for edge in slot.edges.values()]
                ])
            ]
            
            return content
            
        except Exception as e:
            return [dbc.Alert(f"Error: {str(e)}", color="danger")]
    
    elif current_mode == 'perturbation':
        return create_perturbation_panel()
    
    return []

def create_model_summary(edge):
    """Create a compact model summary card."""
    is_default = edge.edge_id == edge.slot_id + "__default" if hasattr(edge, 'slot_id') else False
    
    return html.Div([
        html.Div([
            html.Span(edge.model_type, style={'fontWeight': '500', 'fontSize': '0.85rem'}),
            html.Span(f" (FN≤{edge.fn_constraint*100:.0f}%)", 
                     style={'fontSize': '0.8rem', 'color': '#64748b'})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span(f"AUC: {edge.stats.auc:.3f}", 
                     style={'fontSize': '0.8rem', 'marginRight': '10px'}),
            html.Span(f"FN: {edge.stats.fn_rate:.1%}", 
                     style={'fontSize': '0.8rem', 'marginRight': '10px'}),
            html.Span(f"FP: {edge.stats.fp_rate:.1%}", 
                     style={'fontSize': '0.8rem'})
        ], style={'color': '#64748b'})
    ], style={
        'padding': '8px 12px',
        'backgroundColor': '#f8fafc',
        'border': '1px solid #e2e8f0',
        'borderRadius': '6px',
        'marginBottom': '8px'
    })

def create_perturbation_panel():
    """Create perturbation mode panel."""
    return [
        html.Div([
            html.H5("Perturbation Mode", style={'color': '#1e293b', 'fontWeight': '600', 'marginBottom': '5px'}),
            html.P("Simulate scenarios and see live predictions", 
                   style={'color': '#64748b', 'fontSize': '0.9rem', 'marginBottom': '15px'}),
            dbc.Button("Exit Perturbation", id={'type': 'exit-perturbation-btn', 'index': 0}, 
                      color="secondary", size="sm", className="w-100 mb-3")
        ]),
        
        # Event selector
        html.Div([
            html.H6("Select Event", style={'color': '#1e293b', 'marginBottom': '8px'}),
            dbc.Select(
                id='event-selector',
                options=[
                    {'label': 'CPI Release', 'value': 'CPI'},
                    {'label': 'Employment Report', 'value': 'UNEMPLOYMENT'}
                ],
                value='CPI',
                className="mb-3"
            )
        ]),
        
        # Parameter controls
        html.Div(id='parameter-controls'),
        
        # Live results
        html.Div(id='live-predictions', style={'marginTop': '15px'})
    ]

# Callback: Enter perturbation mode
@app.callback(
    Output('current-mode-store', 'data', allow_duplicate=True),
    [Input('enter-perturbation-btn', 'n_clicks')],
    prevent_initial_call=True
)
def enter_perturbation_mode(enter_clicks):
    """Enter perturbation mode."""
    if enter_clicks:
        return 'perturbation'
    return dash.no_update

# Callback: Exit perturbation mode (separate callback for dynamic button)
@app.callback(
    Output('current-mode-store', 'data', allow_duplicate=True),
    [Input({'type': 'exit-perturbation-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def exit_perturbation_mode(exit_clicks):
    """Exit perturbation mode."""
    if any(exit_clicks):
        return 'default'
    return dash.no_update

# Callback: Update parameter controls based on selected event
@app.callback(
    Output('parameter-controls', 'children'),
    [Input('event-selector', 'value')],
    [State('current-mode-store', 'data')]
)
def update_parameter_controls(selected_event, current_mode):
    """Update parameter controls based on selected event."""
    if current_mode != 'perturbation' or not selected_event:
        return []
    
    # Find relevant edge slots for this event
    relevant_slots = [
        slot for slot in graph.edge_slots.values() 
        if slot.from_node.name == selected_event
    ]
    
    if not relevant_slots:
        return [html.P("No models available for this event", style={'color': '#64748b'})]
    
    # Use first slot for feature specification
    slot = relevant_slots[0]
    feature_spec = slot.feature_spec
    default_edge = slot.get_edge()
    feature_medians = default_edge.feature_medians or {}
    
    controls = []
    
    # Active factor slider
    active_factor = feature_spec['active_factor']
    baseline = feature_medians.get(active_factor, 0.5)
    
    controls.append(html.Div([
        html.H6("Active Factor", style={'color': '#1e293b', 'marginBottom': '8px'}),
        html.Label(active_factor, style={'fontSize': '0.85rem', 'fontWeight': '500'}),
        dcc.Slider(
            id='active-factor-slider',
            min=0,
            max=2.0,
            step=0.1,
            value=baseline,
            marks={i/2: f'{i/2:.1f}' for i in range(0, 5)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        # Preset buttons
        html.Div([
            dbc.ButtonGroup([
                dbc.Button("Small (+0.2)", id='small-shock-btn', size="sm", color="outline-primary"),
                dbc.Button("Medium (+0.5)", id='medium-shock-btn', size="sm", color="outline-warning"),
                dbc.Button("Large (+1.0)", id='large-shock-btn', size="sm", color="outline-danger")
            ], className="w-100")
        ], style={'marginTop': '10px', 'marginBottom': '15px'})
    ]))
    
    # Background factor sliders
    controls.append(html.H6("Background Factors", 
                           style={'color': '#1e293b', 'marginBottom': '8px', 'marginTop': '15px'}))
    
    for i, feature in enumerate(feature_spec['background_features']):
        baseline_bg = feature_medians.get(feature, 0.0)
        controls.append(html.Div([
            html.Label(feature, style={'fontSize': '0.8rem', 'fontWeight': '500'}),
            dcc.Slider(
                id={'type': 'bg-slider', 'index': i},
                min=-2.0,
                max=2.0,
                step=0.05,
                value=baseline_bg,
                marks={-1: '-1', 0: '0', 1: '1'},
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], style={'marginBottom': '10px'}))
    
    return controls

# Callback: Handle preset shock buttons
@app.callback(
    Output('active-factor-slider', 'value'),
    [Input('small-shock-btn', 'n_clicks'),
     Input('medium-shock-btn', 'n_clicks'), 
     Input('large-shock-btn', 'n_clicks')],
    [State('active-factor-slider', 'value'),
     State('event-selector', 'value')],
    prevent_initial_call=True
)
def handle_shock_presets(small_clicks, medium_clicks, large_clicks, current_value, selected_event):
    """Handle preset shock button clicks."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_value
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Get baseline value
    if selected_event:
        relevant_slots = [slot for slot in graph.edge_slots.values() if slot.from_node.name == selected_event]
        if relevant_slots:
            slot = relevant_slots[0]
            default_edge = slot.get_edge()
            baseline = default_edge.feature_medians.get(slot.feature_spec['active_factor'], 0.5)
        else:
            baseline = 0.5
    else:
        baseline = 0.5
    
    if button_id == 'small-shock-btn':
        return baseline + 0.2
    elif button_id == 'medium-shock-btn':
        return baseline + 0.5
    elif button_id == 'large-shock-btn':
        return baseline + 1.0
    
    return current_value

# Callback: Run live predictions
@app.callback(
    [Output('live-predictions', 'children'),
     Output('cytoscape-graph', 'elements')],
    [Input('active-factor-slider', 'value'),
     Input({'type': 'bg-slider', 'index': dash.dependencies.ALL}, 'value')],
    [State('event-selector', 'value'),
     State('current-mode-store', 'data')]
)
def update_live_predictions(active_value, bg_values, selected_event, current_mode):
    """Update live predictions and graph annotations."""
    base_elements = create_graph_elements()
    
    if current_mode != 'perturbation' or not selected_event:
        return [], base_elements
    
    try:
        # Find relevant slots
        relevant_slots = [
            slot for slot in graph.edge_slots.values() 
            if slot.from_node.name == selected_event
        ]
        
        if not relevant_slots:
            return [html.P("No models available", style={'color': '#64748b'})], base_elements
        
        results = []
        updated_elements = []
        
        for element in base_elements:
            # Copy base element
            updated_element = element.copy()
            
            # Add fading to non-involved nodes
            if element['data'].get('type') in ['event', 'instrument']:
                node_name = element['data']['id']
                involved = any(
                    node_name in [slot.from_node.name, slot.to_node.name] 
                    for slot in relevant_slots
                )
                if not involved:
                    if 'classes' not in updated_element:
                        updated_element['classes'] = ''
                    updated_element['classes'] += ' faded'
            
            updated_elements.append(updated_element)
        
        # Run predictions for each relevant slot
        for slot in relevant_slots:
            # Standard feature handling for all slots
            default_edge = slot.get_edge()
            features = {}
            
            # Add active factor
            features[slot.feature_spec['active_factor']] = active_value
            
            # Add background features
            for i, feature in enumerate(slot.feature_spec['background_features']):
                if i < len(bg_values) and bg_values[i] is not None:
                    features[feature] = bg_values[i]
                else:
                    features[feature] = default_edge.feature_medians.get(feature, 0.0)
            
            # Create event context
            event_context = EventContext(
                node=slot.from_node,
                event_date="2025-12-06",
                features=features
            )
            
            # Run prediction
            result = runner.apply(slot.slot_id, event_context)
            
            # Add to results display
            color = "danger" if result.flag_large_move else "success"
            flag_text = "🔴 LARGE MOVE" if result.flag_large_move else "🟢 Normal"
            
            results.append(html.Div([
                html.Div([
                    html.Strong(f"{result.to_node.name}", style={'fontSize': '0.9rem'}),
                    html.Span(f" {flag_text}", style={'marginLeft': '8px', 'fontSize': '0.8rem'})
                ]),
                html.Div([
                    html.Span(f"Prob: {result.prob_large_move:.1%}", 
                             style={'fontSize': '0.8rem', 'fontFamily': 'monospace'})
                ], style={'color': '#64748b'})
            ], style={
                'padding': '8px 12px',
                'backgroundColor': '#fef3c7' if result.flag_large_move else '#f0fdf4',
                'border': f'1px solid {"#fbbf24" if result.flag_large_move else "#bbf7d0"}',
                'borderRadius': '6px',
                'marginBottom': '6px'
            }))
            
            # Add probability annotation to graph nodes
            for element in updated_elements:
                if (element['data'].get('id') == result.to_node.name and 
                    element['data'].get('type') == 'instrument'):
                    # Update label to show probability
                    element['data']['label'] = f"{result.to_node.name}\n{result.prob_large_move:.1%}"
                    if 'classes' not in element:
                        element['classes'] = ''
                    element['classes'] += ' active-prediction'
        
        return results, updated_elements
        
    except Exception as e:
        error_msg = html.Div([
            dbc.Alert(f"Prediction error: {str(e)}", color="danger", dismissable=True)
        ])
        return [error_msg], base_elements

# Update startup script
def update_startup_script():
    """Update startup script."""
    script_content = """#!/bin/bash
# Start the QSIG Macro Graph Explorer webapp (Dash)

echo "🚀 Starting QSIG Macro Graph Explorer (Professional Edition)..."
echo "============================================================"

# Activate virtual environment
source ../webapp_env/bin/activate

# Verify dependencies
echo "📦 Checking dependencies..."
python -c "import dash, dash_cytoscape; print('✅ All dependencies found')" || {
    echo "❌ Missing dependencies. Installing..."
    pip install dash dash-bootstrap-components dash-cytoscape
}

# Start Dash app
echo "🌐 Starting webapp on http://localhost:8050"
echo "   Professional Financial Interface"
echo "   Use Ctrl+C to stop the server"
echo ""

python app.py"""

    with open('start_webapp.sh', 'w') as f:
        f.write(script_content)

if __name__ == '__main__':
    update_startup_script()
    app.run(debug=True, host='0.0.0.0', port=8050)