import React, { createContext, useContext, useState, ReactNode, useCallback, useMemo, useEffect } from 'react';

// ONNX Runtime for browser ML inference
import * as ort from 'onnxruntime-web';

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
    click_count: number;
    hover_count: number;
    blur_count: number;
    focus_count: number;
    wait_count: number;
    close_tab_count: number;
    scroll_depth_max: number;
    interaction_density: number;
    attention_score: number;
    scroll_velocity_avg: number;
    click_to_hover_ratio: number;
    blur_frequency: number;
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
  const [mlPredictor, setMlPredictor] = useState<any>(null);
  const [isMlReady, setIsMlReady] = useState(false);

  // Initialize session start time
  useEffect(() => {
    (window as any).sessionStartTime = Date.now() / 1000;
  }, []);

  // Initialize ML predictor
  useEffect(() => {
    const initializeML = async () => {
      try {
        await initializePredictor();
      } catch (error) {
        console.warn('ML initialization failed, using fallback predictions:', error);
      }
    };

    const initializePredictor = async () => {
      try {
        // Load the ONNX model from backend API
        const response = await fetch('http://localhost:8000/model/interaction_predictor.onnx');
        if (!response.ok) {
          throw new Error('Failed to load ONNX model');
        }

        const modelBuffer = await response.arrayBuffer();
        const session = await ort.InferenceSession.create(modelBuffer);

        setMlPredictor(session);
        setIsMlReady(true);
        console.log('ML predictor initialized successfully');
      } catch (error) {
        console.warn('ONNX model loading failed:', error);
      }
    };

    initializeML();
  }, []);
  const [interactionFeatures, setInteractionFeatures] = useState({
    time_since_last_action: 0,
    avg_time_between_actions: 0,
    session_duration: 0,
    scroll_down_count: 0,
    scroll_up_count: 0,
    click_count: 0,
    hover_count: 0,
    blur_count: 0,
    focus_count: 0,
    wait_count: 0,
    close_tab_count: 0,
    scroll_depth_max: 0,
    interaction_density: 0,
    attention_score: 0,
    scroll_velocity_avg: 0,
    click_to_hover_ratio: 0,
    blur_frequency: 0,
    action_entropy: 0,
    burstiness_score: 0,
    engagement_rhythm: 0
  });

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

  const updatePredictionFromML = useCallback(async () => {
    if (!isMlReady || !mlPredictor) {
      console.warn('ML predictor not ready, using fallback');
      return;
    }

    try {
      // Convert features to array in correct order
      const featureNames = [
        'time_since_last_action', 'avg_time_between_actions', 'session_duration',
        'scroll_down_count', 'scroll_up_count', 'click_count', 'hover_count',
        'blur_count', 'focus_count', 'wait_count', 'close_tab_count',
        'scroll_depth_max', 'interaction_density', 'attention_score',
        'scroll_velocity_avg', 'click_to_hover_ratio', 'blur_frequency',
        'action_entropy', 'burstiness_score', 'engagement_rhythm'
      ];

      const featureArray = featureNames.map(name => interactionFeatures[name as keyof typeof interactionFeatures] || 0);
      const tensor = new ort.Tensor('float32', featureArray, [1, featureArray.length]);

      // Run inference
      const results = await mlPredictor.run({ float_input: tensor });
      const mlProbability = results.output.data[0] * 100; // Convert to percentage

      // Blend ML prediction with current prediction for smooth updates
      const blendedProbability = (currentPrediction.probability * 0.7) + (mlProbability * 0.3);

      updatePrediction({
        ...currentPrediction,
        probability: Math.min(blendedProbability, 95)
      });

      console.log(`ML prediction: ${mlProbability.toFixed(2)}% (blended: ${blendedProbability.toFixed(2)}%)`);
    } catch (error) {
      console.warn('ML prediction error:', error);
    }
  }, [interactionFeatures, currentPrediction, updatePrediction, isMlReady, mlPredictor]);

  const trackInteraction = useCallback((interactionType: string, details?: any) => {
    const now = Date.now() / 1000; // Current time in seconds

    // Update interaction features
    setInteractionFeatures(prev => {
      const updated = { ...prev };

      // Update time-based features
      const timeSinceLast = now - (prev.session_duration || now);
      updated.time_since_last_action = timeSinceLast;
      updated.session_duration = now - ((window as any).sessionStartTime || now);

      switch (interactionType) {
        case 'scroll_down':
          updated.scroll_down_count += 1;
          updated.scroll_depth_max = Math.max(updated.scroll_depth_max, details?.scrollPercent || 0);
          updated.scroll_velocity_avg = (updated.scroll_velocity_avg + (details?.velocity || 0.1)) / 2;
          break;
        case 'scroll_up':
          updated.scroll_up_count += 1;
          updated.scroll_velocity_avg = (updated.scroll_velocity_avg + (details?.velocity || 0.1)) / 2;
          break;
        case 'click':
          updated.click_count += 1;
          break;
        case 'hover':
          updated.hover_count += 1;
          break;
        case 'blur':
          updated.blur_count += 1;
          updated.blur_frequency = updated.blur_count / Math.max(updated.session_duration, 1);
          break;
        case 'focus':
          updated.focus_count += 1;
          break;
        case 'wait':
          updated.wait_count += 1;
          break;
      }

      // Calculate derived features
      const totalActions = updated.scroll_down_count + updated.scroll_up_count +
                          updated.click_count + updated.hover_count +
                          updated.blur_count + updated.focus_count + updated.wait_count;

      updated.interaction_density = totalActions / Math.max(updated.session_duration, 1);

      // Attention score based on focus/blur balance
      const focusBalance = updated.focus_count - updated.blur_count;
      updated.attention_score = Math.max(0, Math.min(1, 0.5 + focusBalance * 0.1 + updated.interaction_density * 0.1));

      // Click to hover ratio
      updated.click_to_hover_ratio = updated.click_count / Math.max(updated.hover_count, 0.1);

      // Simple action entropy (diversity measure)
      const actionTypes = [updated.scroll_down_count, updated.scroll_up_count,
                          updated.click_count, updated.hover_count,
                          updated.blur_count, updated.focus_count, updated.wait_count];
      const total = actionTypes.reduce((a, b) => a + b, 0);
      if (total > 0) {
        const probs = actionTypes.map(count => count / total);
        updated.action_entropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
      }

      return updated;
    });

    // Update local prediction immediately for responsiveness
    let interactionBonus = 0;
    let contextBonus = 0;

    switch (interactionType) {
      case 'scroll_down':
      case 'scroll_up':
        interactionBonus = 3; // Scrolling shows deeper engagement
        contextBonus = details?.scrollPercent * 0.2 || 0;
        break;
      case 'click':
        interactionBonus = 10; // Clicks show strong interest
        break;
      case 'hover':
        interactionBonus = 2; // Hovering shows consideration
        break;
      case 'focus':
        interactionBonus = 5; // Returning focus shows renewed attention
        break;
      case 'blur':
        interactionBonus = -2; // Losing focus reduces engagement
        break;
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

    // Trigger ML prediction update (throttled)
    if (Math.random() < 0.3) { // 30% chance to update ML prediction
      updatePredictionFromML();
    }
  }, [currentPrediction, updatePrediction, updatePredictionFromML]);

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