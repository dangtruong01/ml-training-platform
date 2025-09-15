import React from 'react';
import './App.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import MyData from './components/MyData';
import ProjectDetail from './components/ProjectDetail';
import AutoAnnotation from './components/AutoAnnotation';
import MyTraining from './components/MyTraining';
import MyModel from './components/MyModel';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>ML Training Platform</h1>
          <nav className="nav-container">
            <Link to="/my-data" className="nav-link">📊 My Data</Link>
            <Link to="/auto-annotation" className="nav-link">🤖 Auto-Annotation</Link>
            <Link to="/my-training" className="nav-link">🚀 My Training</Link>
            <Link to="/my-model" className="nav-link">🎯 My Model</Link>
          </nav>
        </header>
        <div className="container">
          <Routes>
            <Route path="/my-data" element={<MyData />} />
            <Route path="/my-data/project/:projectId" element={<ProjectDetail />} />
            <Route path="/auto-annotation" element={<AutoAnnotation />} />
            <Route path="/my-training" element={<MyTraining />} />
            <Route path="/my-model" element={<MyModel />} />
            <Route path="/" element={<MyData />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;