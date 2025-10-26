// File: privads_demo/src/components/PrivAdsProvider.tsx
import React, { createContext, useContext, useState, ReactNode, useCallback, useMemo, useEffect } from 'react';

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
  interactionFeatures: {
    time_since_last_action: number;
    avg_time_between_actions: number;
    session_duration: number;
    scroll_down_count: number;
    scroll_up_count: number;
    scroll_depth_max: number;
    interaction_density: number;
    attention_score: number;
    scroll_velocity_avg: number;
    action_entropy: number;
    burstiness_score: number;
    engagement_rhythm: number;
  };
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

  const [interactionFeatures, setInteractionFeatures] = useState({
    time_since_last_action: 0,
    avg_time_between_actions: 0,
    session_duration: 0,
    scroll_down_count: 0,
    scroll_up_count: 0,
    scroll_depth_max: 0,
    interaction_density: 0,
    attention_score: 0,
    scroll_velocity_avg: 0,
    action_entropy: 0,
    burstiness_score: 0,
    engagement_rhythm: 0
  });

  const [actionHistory, setActionHistory] = useState<any[]>([]);
  const [sessionStartTime] = useState(() => Date.now() / 1000);

  const updatePrediction = useCallback((prediction: AdPrediction) => {
    setCurrentPrediction(prediction);
  }, []);

  const updateSiteContext = useCallback((context: any) => {
    setSiteContext(context);
  }, []);

  const getAdRecommendation = useCallback(async () => {
    try {
      console.log('[PrivAdsProvider] Getting ad recommendation');
      return {
        decision: 'SHOW',
        reason: 'Ready to display personalized ad'
      };
    } catch (error) {
      console.error('Failed to get ad recommendation:', error);
      return {
        decision: 'ERROR',
        reason: 'Unable to get recommendation'
      };
    }
  }, []);

  const searchAds = useCallback(async (query: string) => {
    try {
      console.log('[PrivAdsProvider] Searching ads for:', query);
      return [];
    } catch (error) {
      console.error('Failed to search ads:', error);
      return [];
    }
  }, []);

  const trackInteraction = useCallback((interactionType: string, details?: any) => {
    const now = Date.now() / 1000;
    setActionHistory(prev => [
      ...prev,
      { timestamp: now, action: interactionType, details: details || {} }
    ]);
  }, []);

  // Listen for scroll events
  useEffect(() => {
    function handleScroll(e: Event) {
      const scrollY = window.scrollY || window.pageYOffset;
      const lastScroll = (window as any)._lastScrollY || 0;
      const direction = scrollY > lastScroll ? 'scroll_down' : 'scroll_up';
      const amount = Math.abs(scrollY - lastScroll);
      (window as any)._lastScrollY = scrollY;
      trackInteraction(direction, { scroll_amount: amount / (window.innerHeight || 1) });
    }
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [trackInteraction]);

  // Prune old actions from history
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now() / 1000;
      const windowSize = 10;
      const windowStart = now - windowSize;
      setActionHistory(prev => prev.filter(a => a.timestamp >= windowStart));
    }, 100);
    return () => clearInterval(interval);
  }, []);

  const value = useMemo(() => ({
    userId,
    currentPrediction,
    siteContext,
    updatePrediction,
    updateSiteContext,
    getAdRecommendation,
    searchAds,
    trackInteraction,
    interactionFeatures
  }), [userId, currentPrediction, siteContext, updatePrediction, updateSiteContext, getAdRecommendation, searchAds, trackInteraction, interactionFeatures]);

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