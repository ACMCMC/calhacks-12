import React, { createContext, useContext, useState, ReactNode, useCallback, useMemo } from 'react';

interface AdPrediction {
  probability: number;
  factors: {
    contextRelevance: number;
    userPreferences: number;
    contentType: number;
    interactionHistory: number;
  };
}

interface PrivAdsContextType {
  userId: string;
  currentPrediction: AdPrediction;
  siteContext: any;
  updatePrediction: (prediction: AdPrediction) => void;
  updateSiteContext: (context: any) => void;
  getAdRecommendation: () => Promise<any>;
  searchAds: (query: string) => Promise<any[]>;
  trackInteraction: (interactionType: string, details?: any) => void;
}

const PrivAdsContext = createContext<PrivAdsContextType | undefined>(undefined);

interface PrivAdsProviderProps {
  children: ReactNode;
}

export const PrivAdsProvider: React.FC<PrivAdsProviderProps> = ({ children }) => {
  const [userId] = useState(() => `user_${Math.random().toString(36).substr(2, 9)}`);
  const [currentPrediction, setCurrentPrediction] = useState<AdPrediction>({
    probability: 0,
    factors: {
      contextRelevance: 0,
      userPreferences: 0,
      contentType: 0,
      interactionHistory: 0
    }
  });
  const [siteContext, setSiteContext] = useState({});

  const updatePrediction = useCallback((prediction: AdPrediction) => {
    setCurrentPrediction(prediction);
  }, []);

  const updateSiteContext = useCallback((context: any) => {
    setSiteContext(context);
  }, []);

  const getAdRecommendation = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/get_ad', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          p_receptive: currentPrediction.probability / 100, // Convert to 0-1 scale
          site_context: siteContext
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get ad recommendation:', error);
      return {
        decision: 'ERROR',
        reason: 'Unable to connect to PrivAds API'
      };
    }
  }, [userId, currentPrediction.probability, siteContext]);

  const searchAds = useCallback(async (query: string) => {
    try {
      const response = await fetch('http://localhost:8000/search_ads', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          limit: 5
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result.results || [];
    } catch (error) {
      console.error('Failed to search ads:', error);
      return [];
    }
  }, []);

  const trackInteraction = useCallback((interactionType: string, details?: any) => {
    // Update prediction based on interaction type
    let interactionBonus = 0;
    let contextBonus = 0;

    switch (interactionType) {
      case 'mouse_move':
        interactionBonus = 2; // Mouse movement shows engagement
        break;
      case 'scroll':
        interactionBonus = 5; // Scrolling shows deeper engagement
        contextBonus = details?.scrollPercent * 0.3 || 0;
        break;
      case 'click':
        interactionBonus = 10; // Clicks show strong interest
        break;
      case 'hover':
        interactionBonus = 3; // Hovering shows consideration
        break;
      case 'time_spent':
        interactionBonus = details?.seconds * 0.5 || 0; // Time spent increases receptiveness
        break;
      default:
        interactionBonus = 1;
    }

    const newProbability = Math.min(currentPrediction.probability + interactionBonus, 95);
    const newContextRelevance = Math.min(currentPrediction.factors.contextRelevance + contextBonus, 90);
    const newInteractionHistory = Math.min(currentPrediction.factors.interactionHistory + interactionBonus * 0.8, 85);

    updatePrediction({
      probability: newProbability,
      factors: {
        ...currentPrediction.factors,
        contextRelevance: newContextRelevance,
        interactionHistory: newInteractionHistory
      }
    });
  }, [currentPrediction, updatePrediction]);

  const value = useMemo(() => ({
    userId,
    currentPrediction,
    siteContext,
    updatePrediction,
    updateSiteContext,
    getAdRecommendation,
    searchAds,
    trackInteraction
  }), [userId, currentPrediction, siteContext, updatePrediction, updateSiteContext, getAdRecommendation, searchAds, trackInteraction]);

  return (
    <PrivAdsContext.Provider value={value}>
      {children}
    </PrivAdsContext.Provider>
  );
};

export const usePrivAds = () => {
  const context = useContext(PrivAdsContext);
  if (context === undefined) {
    throw new Error('usePrivAds must be used within a PrivAdsProvider');
  }
  return context;
};

export default PrivAdsProvider;