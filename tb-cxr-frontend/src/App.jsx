import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LandingPage from './LandingPage';
import Dashboard from './Dashboard';
import ContactUs from './ContactUs'; // Import the new ContactUs component
import Navbar from './Navbar';

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/contact" element={<ContactUs />} /> {/* Add route for Contact Us */}
      </Routes>
    </Router>
  );
}

export default App;


