import React from 'react';
import { usePrivAds } from './PrivAdsProvider';
import { useInteractionTracker } from '../hooks/useInteractionTracker';
import { BarChart3, TrendingUp, Eye, MousePointer, Activity } from 'lucide-react';
import './AdPredictionBar.css';

const AdPredictionBar: React.FC = () => {
  const { currentPrediction, interactionFeatures } = usePrivAds();
  const { timeSpent } = useInteractionTracker();

  // Show all input features as bars
  const featureMeta = [
    { key: 'time_since_last_action', name: 'Time Since Last Action', icon: Activity, color: '#6366f1', unit: 's' },
    { key: 'avg_time_between_actions', name: 'Avg Time Between Actions', icon: Activity, color: '#818cf8', unit: 's' },
    { key: 'session_duration', name: 'Session Duration', icon: Activity, color: '#a5b4fc', unit: 's' },
    { key: 'scroll_down_count', name: 'Scroll Down Rate', icon: BarChart3, color: '#3b82f6', unit: '/s' },
    { key: 'scroll_up_count', name: 'Scroll Up Rate', icon: BarChart3, color: '#60a5fa', unit: '/s' },
    { key: 'click_count', name: 'Click Rate', icon: MousePointer, color: '#ef4444', unit: '/s' },
    { key: 'hover_count', name: 'Hover Rate', icon: MousePointer, color: '#f87171', unit: '/s' },
    { key: 'blur_count', name: 'Blur Rate', icon: Eye, color: '#f59e0b', unit: '/s' },
    { key: 'focus_count', name: 'Focus Rate', icon: Eye, color: '#fbbf24', unit: '/s' },
    { key: 'wait_count', name: 'Wait Rate', icon: Activity, color: '#10b981', unit: '/s' },
    { key: 'close_tab_count', name: 'Close Tab Rate', icon: Activity, color: '#14b8a6', unit: '/s' },
    { key: 'scroll_depth_max', name: 'Scroll Depth Max', icon: BarChart3, color: '#06b6d4', unit: '' },
    { key: 'interaction_density', name: 'Interaction Density', icon: Activity, color: '#22d3ee', unit: '/s' },
    { key: 'attention_score', name: 'Attention Score', icon: Eye, color: '#f59e0b', unit: '%' },
    { key: 'scroll_velocity_avg', name: 'Scroll Velocity Avg', icon: BarChart3, color: '#f472b6', unit: '' },
    { key: 'click_to_hover_ratio', name: 'Click/Hover Ratio', icon: MousePointer, color: '#a3e635', unit: '' },
    { key: 'blur_frequency', name: 'Blur Frequency', icon: Eye, color: '#fbbf24', unit: '' },
    { key: 'action_entropy', name: 'Action Entropy', icon: Activity, color: '#facc15', unit: '' },
    { key: 'burstiness_score', name: 'Burstiness Score', icon: Activity, color: '#eab308', unit: '' },
    { key: 'engagement_rhythm', name: 'Engagement Rhythm', icon: Activity, color: '#fde68a', unit: '' },
  ];

  // Helper to safely access feature values
  function getFeatureValue(key: keyof typeof interactionFeatures) {
    if (key === 'attention_score') return interactionFeatures[key] * 100;
    return interactionFeatures[key] || 0;
  }

  const featureBars = featureMeta.map(meta => ({
    ...meta,
    value: getFeatureValue(meta.key as keyof typeof interactionFeatures)
  }));

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

      <div className="prediction-factors" style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
        {featureBars.map((feature, index) => {
          const Icon = feature.icon;
          // For visualization, scale bar width for rates (max reasonable = 1/s, so 100%)
          let barWidth = feature.unit === '%' ? feature.value : Math.min(feature.value * 100, 100);
          return (
            <div key={index} className="factor-item" style={{ flex: '1 1 220px', minWidth: 180, maxWidth: 260 }}>
              <div className="factor-header">
                <Icon size={14} style={{ color: feature.color }} />
                <span className="factor-name">{feature.name}</span>
                <span className="factor-value">{feature.value.toFixed(2)}{feature.unit}</span>
              </div>
              <div className="factor-bar">
                <div
                  className="factor-fill"
                  style={{
                    width: `${barWidth}%`,
                    backgroundColor: feature.color
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