import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/"><span className="icon">🫁</span> TB Detector</Link>
      </div>
      <ul className="navbar-nav">
        <li className="nav-item">
          <Link to="/" className="nav-link">Home</Link>
        </li>
        <li className="nav-item">
          <Link to="/dashboard" className="nav-link">Try Detection</Link>
        </li>
        <li className="nav-item">
          <Link to="/contact" className="nav-link">Contact Us</Link>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar;
