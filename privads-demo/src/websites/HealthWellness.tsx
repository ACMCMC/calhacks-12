import React, { useState, useEffect } from 'react';
import { usePrivAds } from '../components/PrivAdsProvider';
import { Heart, Activity, Moon, Apple, Droplets } from 'lucide-react';
import './HealthWellness.css';

interface WellnessTip {
  id: string;
  title: string;
  category: 'Nutrition' | 'Exercise' | 'Sleep' | 'Mental Health' | 'Hydration';
  content: string;
  estimatedTime: number;
  completed: boolean;
}

const HealthWellness: React.FC = () => {
  const { updateSiteContext, updatePrediction } = usePrivAds();
  const [currentTip, setCurrentTip] = useState<WellnessTip | null>(null);
  const [tipProgress, setTipProgress] = useState(0);
  const [interactions, setInteractions] = useState(0);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const wellnessTips: WellnessTip[] = [
    {
      id: '1',
      title: 'Morning Hydration Ritual',
      category: 'Hydration',
      content: 'Start your day with a large glass of water with lemon. This helps rehydrate your body after sleep and kickstarts your metabolism.',
      estimatedTime: 5,
      completed: false
    },
    {
      id: '2',
      title: 'Mindful Eating Practice',
      category: 'Nutrition',
      content: 'Take time to eat without distractions. Focus on the taste, texture, and aroma of your food to improve digestion and satisfaction.',
      estimatedTime: 15,
      completed: false
    },
    {
      id: '3',
      title: 'Evening Wind-Down Routine',
      category: 'Sleep',
      content: 'Create a calming pre-sleep ritual: dim lights, avoid screens, practice deep breathing, and journal about your day.',
      estimatedTime: 20,
      completed: false
    },
    {
      id: '4',
      title: 'Daily Movement Goals',
      category: 'Exercise',
      content: 'Aim for at least 30 minutes of movement daily. This could be walking, yoga, dancing, or any activity you enjoy.',
      estimatedTime: 30,
      completed: false
    },
    {
      id: '5',
      title: 'Gratitude Meditation',
      category: 'Mental Health',
      content: 'Spend 5 minutes daily reflecting on three things you\'re grateful for. This practice can improve mood and reduce stress.',
      estimatedTime: 10,
      completed: false
    }
  ];

  const categories = [
    { id: 'all', name: 'All Tips', icon: Heart, color: '#ef4444' },
    { id: 'Nutrition', name: 'Nutrition', icon: Apple, color: '#10b981' },
    { id: 'Exercise', name: 'Exercise', icon: Activity, color: '#3b82f6' },
    { id: 'Sleep', name: 'Sleep', icon: Moon, color: '#8b5cf6' },
    { id: 'Mental Health', name: 'Mental Health', icon: Heart, color: '#f59e0b' },
    { id: 'Hydration', name: 'Hydration', icon: Droplets, color: '#06b6d4' }
  ];

  useEffect(() => {
    // Update site context for health & wellness
    updateSiteContext({
      siteType: 'health_wellness',
      contentType: 'wellness_tips',
      currentTip: currentTip?.id || null,
      selectedCategory,
      tipProgress,
      interactions,
      timeSpent: tipProgress * 2, // Mock time spent
      wellnessCategory: currentTip?.category || 'general'
    });

    // Update prediction based on wellness engagement
    const basePrediction = 40; // Health content generally receptive
    const progressBonus = tipProgress * 0.3; // More engaged = more receptive
    const interactionBonus = interactions * 3; // Interactive elements increase receptiveness
    const categoryBonus = currentTip?.category === 'Mental Health' ? 25 : 0; // Mental health content highly receptive

    updatePrediction({
      probability: Math.min(basePrediction + progressBonus + interactionBonus + categoryBonus, 95),
      factors: {
        contextRelevance: 75 + (tipProgress * 0.2),
        userPreferences: 70 + categoryBonus,
        contentType: 85, // Health content very receptive
        interactionHistory: tipProgress * 0.5 + (interactions * 0.8)
      }
    });
  }, [currentTip, tipProgress, interactions, selectedCategory, updateSiteContext, updatePrediction]);

  const handleTipSelect = (tip: WellnessTip) => {
    setCurrentTip(tip);
    setTipProgress(0);
    setInteractions(0);
  };

  const handleInteraction = () => {
    setInteractions(prev => prev + 1);
  };

  const handleProgressUpdate = (progress: number) => {
    setTipProgress(progress);
  };

  const filteredTips = selectedCategory === 'all'
    ? wellnessTips
    : wellnessTips.filter(tip => tip.category === selectedCategory);

  const getCategoryIcon = (categoryName: string) => {
    const category = categories.find(c => c.name === categoryName);
    return category ? category.icon : Heart;
  };

  const getCategoryColor = (categoryName: string) => {
    const category = categories.find(c => c.name === categoryName);
    return category ? category.color : '#6b7280';
  };

  return (
    <div className="health-wellness">
      <div className="site-header">
        <div className="header-content">
          <Heart size={48} />
          <div>
            <h1>VitaLife Hub</h1>
            <p>Your personal wellness companion for a healthier life</p>
          </div>
        </div>
      </div>

      <div className="content-layout">
        <div className="sidebar">
          <div className="category-filter">
            <h3>Wellness Categories</h3>
            <div className="category-buttons">
              {categories.map(category => {
                const Icon = category.icon;
                return (
                  <button
                    key={category.id}
                    className={`category-btn ${selectedCategory === category.id ? 'active' : ''}`}
                    onClick={() => setSelectedCategory(category.id)}
                    style={{
                      borderColor: selectedCategory === category.id ? category.color : 'transparent'
                    }}
                  >
                    <Icon size={16} style={{ color: category.color }} />
                    {category.name}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="tips-list">
            <h3>Daily Wellness Tips</h3>
            {filteredTips.map(tip => {
              const Icon = getCategoryIcon(tip.category);
              return (
                <div
                  key={tip.id}
                  className={`tip-card ${currentTip?.id === tip.id ? 'active' : ''}`}
                  onClick={() => handleTipSelect(tip)}
                >
                  <div className="tip-header">
                    <Icon size={18} style={{ color: getCategoryColor(tip.category) }} />
                    <span className="tip-category">{tip.category}</span>
                    <span className="tip-time">{tip.estimatedTime} min</span>
                  </div>
                  <h4>{tip.title}</h4>
                  <p>{tip.content.substring(0, 80)}...</p>
                  {tip.completed && (
                    <div className="completion-indicator">✓ Completed</div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="main-content">
          {currentTip ? (
            <div className="tip-detail">
              <div className="tip-header">
                <div className="tip-meta">
                  <span
                    className="category-badge"
                    style={{ backgroundColor: getCategoryColor(currentTip.category) }}
                  >
                    {currentTip.category}
                  </span>
                  <h1>{currentTip.title}</h1>
                  <p className="tip-description">{currentTip.content}</p>
                </div>
                <div className="tip-stats">
                  <div className="stat">
                    <Activity size={16} />
                    <span>{interactions} interactions</span>
                  </div>
                  <div className="stat">
                    <Heart size={16} />
                    <span>{tipProgress}% complete</span>
                  </div>
                </div>
              </div>

              <div className="tip-content">
                <div className="interactive-section">
                  <h3>Practice This Tip</h3>

                  {currentTip.category === 'Hydration' && (
                    <div className="hydration-tracker">
                      <h4>Water Intake Tracker</h4>
                      <div className="water-glasses">
                        {Array.from({ length: 8 }, (_, i) => (
                          <div
                            key={i}
                            className={`water-glass ${i < Math.floor(interactions / 2) ? 'filled' : ''}`}
                            onClick={handleInteraction}
                          >
                            <Droplets size={24} />
                          </div>
                        ))}
                      </div>
                      <p>Click glasses to track your water intake!</p>
                      <button
                        className="practice-btn"
                        onClick={() => handleProgressUpdate(Math.min(tipProgress + 25, 100))}
                      >
                        Mark as Practiced
                      </button>
                    </div>
                  )}

                  {currentTip.category === 'Exercise' && (
                    <div className="exercise-tracker">
                      <h4>Movement Minutes</h4>
                      <div className="activity-buttons">
                        <button
                          className="activity-btn"
                          onClick={() => {
                            handleInteraction();
                            handleProgressUpdate(Math.min(tipProgress + 10, 100));
                          }}
                        >
                          🏃‍♀️ Walking
                        </button>
                        <button
                          className="activity-btn"
                          onClick={() => {
                            handleInteraction();
                            handleProgressUpdate(Math.min(tipProgress + 15, 100));
                          }}
                        >
                          🧘 Yoga
                        </button>
                        <button
                          className="activity-btn"
                          onClick={() => {
                            handleInteraction();
                            handleProgressUpdate(Math.min(tipProgress + 20, 100));
                          }}
                        >
                          🏊‍♀️ Swimming
                        </button>
                      </div>
                      <p>Log your daily movement activities!</p>
                    </div>
                  )}

                  {currentTip.category === 'Sleep' && (
                    <div className="sleep-tracker">
                      <h4>Sleep Quality Journal</h4>
                      <div className="sleep-rating">
                        {[1, 2, 3, 4, 5].map(rating => (
                          <button
                            key={rating}
                            className={`rating-star ${interactions >= rating ? 'active' : ''}`}
                            onClick={() => {
                              setInteractions(rating);
                              handleProgressUpdate(Math.min(tipProgress + 20, 100));
                            }}
                          >
                            ⭐
                          </button>
                        ))}
                      </div>
                      <p>Rate your sleep quality tonight</p>
                    </div>
                  )}

                  {currentTip.category === 'Nutrition' && (
                    <div className="nutrition-log">
                      <h4>Mindful Eating Log</h4>
                      <div className="meal-tracker">
                        <button
                          className="meal-btn"
                          onClick={() => {
                            handleInteraction();
                            handleProgressUpdate(Math.min(tipProgress + 15, 100));
                          }}
                        >
                          🍎 Breakfast
                        </button>
                        <button
                          className="meal-btn"
                          onClick={() => {
                            handleInteraction();
                            handleProgressUpdate(Math.min(tipProgress + 15, 100));
                          }}
                        >
                          🥗 Lunch
                        </button>
                        <button
                          className="meal-btn"
                          onClick={() => {
                            handleInteraction();
                            handleProgressUpdate(Math.min(tipProgress + 15, 100));
                          }}
                        >
                          🍽️ Dinner
                        </button>
                      </div>
                      <p>Log your mindful eating sessions</p>
                    </div>
                  )}

                  {currentTip.category === 'Mental Health' && (
                    <div className="gratitude-journal">
                      <h4>Gratitude Practice</h4>
                      <textarea
                        placeholder="Write three things you're grateful for today..."
                        className="gratitude-input"
                        onChange={handleInteraction}
                      />
                      <button
                        className="practice-btn"
                        onClick={() => handleProgressUpdate(Math.min(tipProgress + 30, 100))}
                      >
                        Save Gratitude
                      </button>
                      <p>Practice gratitude daily for better mental wellness</p>
                    </div>
                  )}
                </div>

                <div className="ad-placeholder">
                  <div className="ad-content">
                    <h4>Personalized Wellness Plans</h4>
                    <p>Get AI-powered wellness recommendations tailored to your lifestyle and goals.</p>
                    <button className="ad-button">Explore Wellness AI</button>
                  </div>
                </div>

                <div className="benefits-section">
                  <h3>Benefits of This Practice</h3>
                  <div className="benefits-grid">
                    <div className="benefit">
                      <div className="benefit-icon">🧠</div>
                      <h4>Improved Focus</h4>
                      <p>Regular practice enhances mental clarity and concentration.</p>
                    </div>
                    <div className="benefit">
                      <div className="benefit-icon">💪</div>
                      <h4>Increased Energy</h4>
                      <p>Consistent habits lead to better physical and mental energy.</p>
                    </div>
                    <div className="benefit">
                      <div className="benefit-icon">😊</div>
                      <h4>Reduced Stress</h4>
                      <p>Mindful practices help manage daily stress and anxiety.</p>
                    </div>
                  </div>
                </div>

                <div className="progress-tracker">
                  <h3>Your Progress</h3>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${tipProgress}%` }}
                    />
                  </div>
                  <p>{tipProgress}% complete • {interactions} interactions logged</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="welcome-screen">
              <Heart size={64} />
              <h2>Welcome to VitaLife Hub</h2>
              <p>Discover personalized wellness tips and practices to improve your health and happiness. Select a tip from the sidebar to begin your wellness journey.</p>
              <div className="quick-stats">
                <div className="stat-card">
                  <Activity size={24} />
                  <span>5 Tips Available</span>
                </div>
                <div className="stat-card">
                  <Heart size={24} />
                  <span>Multiple Categories</span>
                </div>
                <div className="stat-card">
                  <Moon size={24} />
                  <span>Daily Practices</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HealthWellness;