import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import './App.css';

// Import website components
import NewsSite from './websites/NewsSite';
import ChessTutorial from './websites/ChessTutorial';
import HealthWellness from './websites/HealthWellness';
import EducationalPlatform from './websites/EducationalPlatform';

// Import website styles
import './websites/NewsSite.css';
import './websites/ChessTutorial.css';
import './websites/HealthWellness.css';
import './websites/EducationalPlatform.css';

// Import shared components
import AdPredictionBar from './components/AdPredictionBar';
import PrivAdsProvider from './components/PrivAdsProvider';

// Import icons
import { Newspaper, BookOpen, Heart, GraduationCap, Home } from 'lucide-react';

// Navigation component
function Navigation() {
  const location = useLocation();

  const websites = [
    {
      id: 'news',
      name: 'TruthLens News',
      icon: Newspaper,
      path: '/news',
      description: 'Privacy-first news aggregator'
    },
    {
      id: 'chess',
      name: 'Chess Master',
      icon: BookOpen,
      path: '/chess',
      description: 'Interactive chess tutorials'
    },
    {
      id: 'health',
      name: 'VitaLife Hub',
      icon: Heart,
      path: '/health',
      description: 'Health & wellness platform'
    },
    {
      id: 'education',
      name: 'Learnify',
      icon: GraduationCap,
      path: '/education',
      description: 'Educational platform'
    }
  ];

  return (
    <nav className="site-navigation">
      <div className="nav-container">
        <Link to="/" className="nav-brand">
          <Home size={20} />
          <span>PrivAds Demo</span>
        </Link>
        <div className="nav-links">
          {websites.map(site => {
            const Icon = site.icon;
            const isActive = location.pathname === site.path;
            return (
              <Link
                key={site.id}
                to={site.path}
                className={`nav-link ${isActive ? 'active' : ''}`}
              >
                <Icon size={16} />
                {site.name}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}

// Home page component
function HomePage() {
  const websites = [
    {
      id: 'news',
      name: 'TruthLens News',
      icon: Newspaper,
      path: '/news',
      description: 'Privacy-first news aggregator',
      color: 'from-blue-600 to-blue-800'
    },
    {
      id: 'chess',
      name: 'Chess Master',
      icon: BookOpen,
      path: '/chess',
      description: 'Interactive chess tutorials',
      color: 'from-amber-600 to-amber-800'
    },
    {
      id: 'health',
      name: 'VitaLife Hub',
      icon: Heart,
      path: '/health',
      description: 'Health & wellness platform',
      color: 'from-green-600 to-green-800'
    },
    {
      id: 'education',
      name: 'Learnify',
      icon: GraduationCap,
      path: '/education',
      description: 'Educational platform',
      color: 'from-purple-600 to-purple-800'
    }
  ];

  return (
    <div className="home-page">
      <div className="hero">
        <h1>PrivAds Demo Websites</h1>
        <p>Experience privacy-first advertising in action. Each website demonstrates how PrivAds serves personalized ads without tracking your behavior.</p>
      </div>

      <div className="websites-grid">
        {websites.map(site => {
          const Icon = site.icon;
          return (
            <Link key={site.id} to={site.path} className="website-card">
              <div className={`website-icon bg-gradient-to-br ${site.color}`}>
                <Icon size={48} />
              </div>
              <h3>{site.name}</h3>
              <p>{site.description}</p>
              <span className="explore-link">Explore →</span>
            </Link>
          );
        })}
      </div>

      <div className="demo-info">
        <h2>How It Works</h2>
        <div className="info-grid">
          <div className="info-item">
            <h4>🔒 Privacy First</h4>
            <p>Ads are served based only on your current context, not your browsing history.</p>
          </div>
          <div className="info-item">
            <h4>🎯 Real-time Generation</h4>
            <p>Ad content is generated live using local AI models in your browser.</p>
          </div>
          <div className="info-item">
            <h4>📊 Live Predictions</h4>
            <p>See real-time ad interaction predictions as you interact with each site.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <PrivAdsProvider>
      <Router>
        <div className="App">
          <AdPredictionBar />
          <Navigation />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/news" element={<NewsSite />} />
              <Route path="/chess" element={<ChessTutorial />} />
              <Route path="/health" element={<HealthWellness />} />
              <Route path="/education" element={<EducationalPlatform />} />
            </Routes>
          </main>
        </div>
      </Router>
    </PrivAdsProvider>
  );
}

export default App;
