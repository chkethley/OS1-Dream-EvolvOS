import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Cpu, Database, Brain, GitMerge, Search } from 'lucide-react';

const EvolvOSDashboard = () => {
  const [systemState, setSystemState] = useState({
    memory: { usage: 0, items: 0 },
    evolution: { cycles: 0, improvements: 0 },
    performance: []
  });
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');

  // Simulated data for demo
  useEffect(() => {
    // This would be replaced with actual API calls
    setSystemState({
      memory: { usage: 45, items: 124 },
      evolution: { cycles: 8, improvements: 14 },
      performance: [
        { name: 'Cycle 1', accuracy: 0.72, efficiency: 0.65, memory_usage: 0.55 },
        { name: 'Cycle 2', accuracy: 0.74, efficiency: 0.68, memory_usage: 0.53 },
        { name: 'Cycle 3', accuracy: 0.78, efficiency: 0.72, memory_usage: 0.50 },
        { name: 'Cycle 4', accuracy: 0.79, efficiency: 0.75, memory_usage: 0.47 },
        { name: 'Cycle 5', accuracy: 0.81, efficiency: 0.77, memory_usage: 0.42 },
      ]
    });
  }, []);

  const handleQuery = async () => {
    // This would call the API endpoint
    // const response = await fetch('/api/query', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ query })
    // });
    // const data = await response.json();
    // setQueryResult(data.result);
    
    // Simulated response for demo
    setQueryResult({
      content: "The memory system architecture combines volatile and archival storage with entity relationship tracking for optimal recall and retrieval efficiency.",
      score: 0.92,
      source: "system_documentation"
    });
  };

  const triggerEvolution = async () => {
    // This would trigger an evolution cycle via API
    alert("Evolution cycle initiated!");
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="bg-blue-600 text-white p-4">
        <h1 className="text-2xl font-bold">OS1-Dream-EvolvOS Dashboard</h1>
      </header>
      
      <nav className="bg-gray-800 text-white">
        <ul className="flex">
          <li className={`p-4 cursor-pointer ${activeTab === 'dashboard' ? 'bg-gray-700' : ''}`} 
              onClick={() => setActiveTab('dashboard')}>
            <Cpu className="inline mr-2" size={16} />Dashboard
          </li>
          <li className={`p-4 cursor-pointer ${activeTab === 'memory' ? 'bg-gray-700' : ''}`}
              onClick={() => setActiveTab('memory')}>
            <Database className="inline mr-2" size={16} />Memory
          </li>
          <li className={`p-4 cursor-pointer ${activeTab === 'evolution' ? 'bg-gray-700' : ''}`}
              onClick={() => setActiveTab('evolution')}>
            <Brain className="inline mr-2" size={16} />Evolution
          </li>
        </ul>
      </nav>
      
      <main className="flex-1 p-6 overflow-auto">
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded shadow">
              <h2 className="text-xl font-semibold mb-4">System Overview</h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded">
                  <h3 className="text-lg font-medium">Memory Usage</h3>
                  <p className="text-3xl font-bold">{systemState.memory.usage}%</p>
                  <p className="text-gray-500">{systemState.memory.items} items stored</p>
                </div>
                <div className="bg-green-50 p-4 rounded">
                  <h3 className="text-lg font-medium">Evolution</h3>
                  <p className="text-3xl font-bold">{systemState.evolution.cycles}</p>
                  <p className="text-gray-500">{systemState.evolution.improvements} improvements</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white p-6 rounded shadow">
              <h2 className="text-xl font-semibold mb-4">Performance Metrics</h2>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={systemState.performance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#8884d8" />
                  <Line type="monotone" dataKey="efficiency" stroke="#82ca9d" />
                  <Line type="monotone" dataKey="memory_usage" stroke="#ffc658" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div className="bg-white p-6 rounded shadow md:col-span-2">
              <h2 className="text-xl font-semibold mb-4">Query System</h2>
              <div className="flex mb-4">
                <input 
                  type="text" 
                  className="flex-1 p-2 border border-gray-300 rounded-l"
                  placeholder="Enter your query..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <button 
                  className="bg-blue-500 text-white px-4 py-2 rounded-r"
                  onClick={handleQuery}
                >
                  <Search className="inline" size={16} />
                </button>
              </div>
              
              {queryResult && (
                <div className="bg-gray-50 p-4 rounded">
                  <h3 className="font-medium">Result (Score: {queryResult.score})</h3>
                  <p className="mt-2">{queryResult.content}</p>
                  <p className="text-gray-500 text-sm mt-2">Source: {queryResult.source}</p>
                </div>
              )}
            </div>
          </div>
        )}
        
        {activeTab === 'memory' && (
          <div className="bg-white p-6 rounded shadow">
            <h2 className="text-xl font-semibold mb-4">Memory Management</h2>
            <p>This panel would allow you to explore the hierarchical memory system, view stored entities, and manage memory contents.</p>
          </div>
        )}
        
        {activeTab === 'evolution' && (
          <div className="bg-white p-6 rounded shadow">
            <h2 className="text-xl font-semibold mb-4">System Evolution</h2>
            <p className="mb-4">Trigger an evolution cycle to improve system performance.</p>
            <button 
              className="bg-green-500 text-white px-4 py-2 rounded flex items-center"
              onClick={triggerEvolution}
            >
              <GitMerge className="mr-2" size={16} />
              Start Evolution Cycle
            </button>
          </div>
        )}
      </main>
      
      <footer className="bg-gray-200 p-4 text-center text-gray-600">
        OS1-Dream-EvolvOS Self-Evolving AI System
      </footer>
    </div>
  );
};

export default EvolvOSDashboard;
