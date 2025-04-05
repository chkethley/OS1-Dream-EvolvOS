from flask import Flask, request, jsonify
import os
import sys
import json
import time

# Add the OS1-Dream-EvolvOS directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'OS1-Dream-EvolvOS')))

# Import system components
try:
    from OS1_Dream_EvolvOS.Meta_data.main import EvolvOS
except ImportError as e:
    print(f"Error importing EvolvOS: {e}")
    print("Make sure all required modules are in the correct directories.")
    sys.exit(1)

app = Flask(__name__)

# Initialize the system
system = EvolvOS()
system.initialize()

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    status = system.get_status()
    return jsonify(status)

@app.route('/api/query', methods=['POST'])
def query_system():
    """Query the memory system"""
    data = request.json
    query = data.get('query', '')
    strategy = data.get('strategy', 'hybrid')
    top_k = data.get('top_k', 5)
    
    # Process query
    results = system.retrieve(query, strategy=strategy, top_k=top_k)
    
    return jsonify({
        'query': query,
        'results': results,
        'timestamp': time.time()
    })

@app.route('/api/store', methods=['POST'])
def store_memory():
    """Store content in memory"""
    data = request.json
    content = data.get('content', '')
    metadata = data.get('metadata', {})
    
    # Store in memory
    memory_id = system.store_memory(content, metadata)
    
    return jsonify({
        'memory_id': memory_id,
        'content': content,
        'metadata': metadata,
        'timestamp': time.time()
    })

@app.route('/api/evolve', methods=['POST'])
def evolve_system():
    """Trigger system evolution"""
    data = request.json
    component = data.get('component', None)
    
    # Trigger evolution
    system.evolve_system(component)
    
    return jsonify({
        'status': 'evolution_triggered',
        'component': component,
        'timestamp': time.time()
    })

@app.route('/api/save', methods=['POST'])
def save_state():
    """Save system state"""
    data = request.json
    path = data.get('path', 'evolvos_state')
    
    # Save state
    system.save_state(path)
    
    return jsonify({
        'status': 'state_saved',
        'path': path,
        'timestamp': time.time()
    })

@app.route('/api/load', methods=['POST'])
def load_state():
    """Load system state"""
    data = request.json
    path = data.get('path', '')
    
    if not path:
        return jsonify({
            'status': 'error',
            'message': 'No path provided'
        }), 400
    
    # Load state
    success = system.load_state(path)
    
    return jsonify({
        'status': 'state_loaded' if success else 'error',
        'path': path,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    app.run(debug=True)
