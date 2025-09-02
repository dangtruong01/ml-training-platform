import React from 'react';
import './App.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Train from './components/Train';
import Predict from './components/Predict';
import Annotate from './components/Annotate';
import AutoAnnotation from './components/AutoAnnotation';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Object Detection & Segmentation Platform</h1>
          <nav className="nav-container">
            <Link to="/annotate" className="nav-link">📝 Annotate</Link>
            <Link to="/auto-annotation" className="nav-link">🤖 Auto-Annotation</Link>
            <Link to="/train" className="nav-link">🎯 Train</Link>
            <Link to="/predict" className="nav-link">🔍 Predict</Link>
          </nav>
        </header>
        <div className="container">
          <Routes>
            <Route path="/train" element={<Train />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/annotate" element={<Annotate />} />
            <Route path="/auto-annotation" element={<AutoAnnotation />} />
            <Route path="/" element={<Predict />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;