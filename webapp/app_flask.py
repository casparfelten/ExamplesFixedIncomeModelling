#!/usr/bin/env python3
"""
QSIG Macro Graph Explorer - Flask Edition
Clean, minimal interface for Bayesian graph models
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify
from pathlib import Path

# Add project source to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from graph.registry import load_graph
    from graph.edge_runner import EdgeRunner
    from graph.types import EventContext
except ImportError as e:
    print(f"Failed to import graph modules: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print(f"Source path: {project_root / 'src'}")
    raise

app = Flask(__name__)

# Load graph and runner globally
graph_path = project_root / 'registry' / 'macro_graph.json'
models_path = project_root / 'registry' / 'models'

graph = None
runner = None

def init_graph():
    """Initialize graph and runner on first request"""
    global graph, runner
    if graph is None:
        graph = load_graph(str(graph_path))
        runner = EdgeRunner(graph, str(models_path))

@app.route('/')
def index():
    """Main page with graph overview"""
    init_graph()
    
    # Prepare graph data for visualization
    nodes = []
    edges = []
    
    # Add event nodes (left side)
    for node_id, node in graph.nodes.items():
        if node.type == 'event':
            nodes.append({
                'id': node_id,
                'name': node.name,
                'type': 'event',
                'description': node.description,
                'x': 100,
                'y': 100 + len([n for n in nodes if n['type'] == 'event']) * 120
            })
    
    # Add instrument nodes (right side)
    for node_id, node in graph.nodes.items():
        if node.type == 'instrument':
            nodes.append({
                'id': node_id,
                'name': node.name,
                'type': 'instrument', 
                'description': node.description,
                'x': 400,
                'y': 100 + len([n for n in nodes if n['type'] == 'instrument']) * 80
            })
    
    # Add edges between nodes
    for slot_id, slot in graph.edge_slots.items():
        edges.append({
            'id': slot_id,
            'from': slot.from_node.name,
            'to': slot.to_node.name,
            'description': slot.description,
            'active_factor': slot.feature_spec['active_factor'],
            'background_features': slot.feature_spec['background_features']
        })
    
    return render_template('index.html', nodes=nodes, edges=edges)

@app.route('/slot/<slot_id>')
def slot_details(slot_id):
    """Show detailed information about an edge slot"""
    init_graph()
    
    try:
        slot = graph.get_slot(slot_id)
        default_edge = slot.get_edge()
        
        slot_data = {
            'slot_id': slot_id,
            'from_node': slot.from_node.name,
            'to_node': slot.to_node.name,
            'description': slot.description,
            'active_factor': slot.feature_spec['active_factor'],
            'background_features': slot.feature_spec['background_features'],
            'feature_medians': default_edge.feature_medians or {},
            'model_type': default_edge.model_type,
            'fn_constraint': default_edge.fn_constraint,
            'stats': {
                'auc': default_edge.stats.auc,
                'fn_rate': default_edge.stats.fn_rate,
                'fp_rate': default_edge.stats.fp_rate,
                'base_rate': default_edge.stats.base_rate,
                'tp': default_edge.stats.tp,
                'fp': default_edge.stats.fp,
                'fn': default_edge.stats.fn,
                'tn': default_edge.stats.tn
            }
        }
        
        return render_template('slot_details.html', slot=slot_data)
    
    except Exception as e:
        return f"Error loading slot {slot_id}: {str(e)}", 404

@app.route('/perturbation/<event_type>')
def perturbation(event_type):
    """Parameter perturbation interface"""
    init_graph()
    
    # Find all slots that originate from this event type
    relevant_slots = []
    for slot_id, slot in graph.edge_slots.items():
        if slot.from_node.name == event_type:
            default_edge = slot.get_edge()
            relevant_slots.append({
                'slot_id': slot_id,
                'to_node': slot.to_node.name,
                'active_factor': slot.feature_spec['active_factor'],
                'background_features': slot.feature_spec['background_features'],
                'feature_medians': default_edge.feature_medians or {}
            })
    
    if not relevant_slots:
        return f"No edges found for event type: {event_type}", 404
    
    return render_template('perturbation_v2.html', 
                         event_type=event_type, 
                         slots=relevant_slots,
                         first_slot=relevant_slots[0])

@app.route('/api/predict', methods=['POST'])
def predict():
    """Run predictions for given parameters"""
    init_graph()
    
    try:
        data = request.get_json()
        event_type = data['event_type']
        active_value = float(data['active_value'])
        background_values = [float(v) if v != '' else None for v in data['background_values']]
        
        results = []
        
        # Find relevant slots
        for slot_id, slot in graph.edge_slots.items():
            if slot.from_node.name == event_type:
                default_edge = slot.get_edge()
                features = {}
                
                # Add active factor
                features[slot.feature_spec['active_factor']] = active_value
                
                # Add background features
                for i, feature in enumerate(slot.feature_spec['background_features']):
                    if i < len(background_values) and background_values[i] is not None:
                        features[feature] = background_values[i]
                    else:
                        features[feature] = default_edge.feature_medians.get(feature, 0.0)
                
                # Create event context and run prediction
                event_context = EventContext(
                    node=slot.from_node,
                    event_date="2025-12-06",
                    features=features
                )
                
                result = runner.apply(slot_id, event_context)
                
                results.append({
                    'slot_id': slot_id,
                    'to_node': result.to_node.name,
                    'probability': round(result.prob_large_move, 4),
                    'flag_large_move': result.flag_large_move,
                    'threshold': result.large_move_threshold
                })
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)