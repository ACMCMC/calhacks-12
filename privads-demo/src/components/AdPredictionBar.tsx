import React from 'react';
import { usePrivAds } from './PrivAdsProvider';
import { useInteractionTracker } from '../hooks/useInteractionTracker';
import { BarChart3, TrendingUp, Eye, MousePointer, Activity } from 'lucide-react';
import './AdPredictionBar.css';

const AdPredictionBar: React.FC = () => {
  const { currentPrediction } = usePrivAds();
  const { timeSpent } = useInteractionTracker();

  const factors = [
    {
      name: 'Context Relevance',
      value: currentPrediction.factors.contextRelevance,
      icon: BarChart3,
      color: '#3b82f6'
    },
    {
      name: 'User Preferences',
      value: currentPrediction.factors.userPreferences,
      icon: TrendingUp,
      color: '#10b981'
    },
    {
      name: 'Content Type',
      value: currentPrediction.factors.contentType,
      icon: Eye,
      color: '#f59e0b'
    },
    {
      name: 'Interaction History',
      value: currentPrediction.factors.interactionHistory,
      icon: MousePointer,
      color: '#ef4444'
    }
  ];

  return (
    <div className="ad-prediction-bar">
      <div className="prediction-header">
        <div className="prediction-main">
          <div className="prediction-score">
            <div className="score-circle">
              <span className="score-value">{Math.round(currentPrediction.probability)}%</span>
              <span className="score-label">Ad Receptiveness</span>
            </div>
          </div>
          <div className="prediction-meta">
            <div className="meta-item">
              <Activity size={14} />
              <span>{timeSpent}s active</span>
            </div>
            <div className="meta-item">
              <MousePointer size={14} />
              <span>Live tracking</span>
            </div>
          </div>
        </div>
      </div>

      <div className="prediction-factors">
        {factors.map((factor, index) => {
          const Icon = factor.icon;
          return (
            <div key={index} className="factor-item">
              <div className="factor-header">
                <Icon size={14} style={{ color: factor.color }} />
                <span className="factor-name">{factor.name}</span>
                <span className="factor-value">{Math.round(factor.value)}%</span>
              </div>
              <div className="factor-bar">
                <div
                  className="factor-fill"
                  style={{
                    width: `${factor.value}%`,
                    backgroundColor: factor.color
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="prediction-footer">
        <span className="live-indicator">
          <div className="pulse-dot"></div>
          Live AI Analysis
        </span>
      </div>
    </div>
  );
};

export default AdPredictionBar;