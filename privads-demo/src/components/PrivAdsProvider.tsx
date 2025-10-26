import React, { createContext, useContext, useState, ReactNode, useCallback, useMemo, useEffect } from 'react';

// ONNX Runtime for browser ML inference (loaded globally from CDN)
declare global {
  interface Window {
    ort: any;
  }
}

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
        // Load ONNX model from backend API (avoids CORS and MIME-type issues)
        const modelUrl = 'http://localhost:8000/model/interaction_predictor.onnx';        const response = await fetch(modelUrl);
        if (!response.ok) {
          throw new Error(`Failed to load ONNX model from ${modelUrl} (status=${response.status})`);
        }

        const modelBuffer = await response.arrayBuffer();        // Create session using the CPU execution provider to avoid loading the
        // wasm backend (which requires a .wasm runtime file and correct MIME types).
        const session = await window.ort.InferenceSession.create(modelBuffer, {
          executionProviders: ['cpu'],
          graphOptimizationLevel: 'all'
        } as any);

        setMlPredictor(session);
        setIsMlReady(true);
        console.log('ML predictor initialized successfully (cpu backend)');
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
    scroll_depth_max: 0,
    interaction_density: 0,
    attention_score: 0,
    scroll_velocity_avg: 0,
    action_entropy: 0,
    burstiness_score: 0,
    engagement_rhythm: 0
  });

  // --- Action history state (for proper sliding window) ---
  const [actionHistory, setActionHistory] = useState<UserAction[]>([]);

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
      // ML predictor not ready: do nothing (no fallback, no error)
      return;
    }

    try {
      // Convert features to array in correct order
      const featureNames = [
        'time_since_last_action', 'avg_time_between_actions', 'session_duration',
        'scroll_down_count', 'scroll_up_count', 'scroll_depth_max',
        'interaction_density', 'attention_score', 'scroll_velocity_avg',
        'action_entropy', 'burstiness_score', 'engagement_rhythm'
      ];

      const featureArray = featureNames.map(name => interactionFeatures[name as keyof typeof interactionFeatures] || 0);
      const tensor = new window.ort.Tensor('float32', featureArray, [1, featureArray.length]);

      // Run inference
      const results = await mlPredictor.run({ float_input: tensor });
      // Log the output keys for debugging
      const outputKeys = Object.keys(results);
      console.log('ONNX output keys:', outputKeys);
      const firstKey = outputKeys[0];

      // Use the correct output key and ensure conversion to Number
      // 'probabilities' is an object with a .data property (Float32Array)
      const probabilities = results['probabilities'];
      const probArray = Array.from(probabilities.data).map(Number);
      // Defensive: check for NaN or undefined
      const mlProbability = (probArray[1] !== undefined && !isNaN(probArray[1])) ? probArray[1] * 100 : 0; // Use positive class probability, fallback to 0

      // Debug: print input features and ONNX output
      console.log('ML input features:', featureArray);
      console.log('ONNX output probabilities:', probArray);

      // Print model input names for debugging
      if (mlPredictor && mlPredictor.inputNames) {
        console.log('ONNX model input names:', mlPredictor.inputNames);
      }
      // Print raw ONNX output
      console.log('ONNX raw output:', results);

      // Update prediction with ML probability only
      updatePrediction({
        ...currentPrediction,
        probability: Math.min(mlProbability, 95)
      });

      console.log(`ML prediction: ${mlProbability.toFixed(2)}%`);
    } catch (error) {
      console.warn('ML prediction error:', error);
    }
  }, [interactionFeatures, currentPrediction, updatePrediction, isMlReady, mlPredictor]);

  // --- Real-time sliding window update ---
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now() / 1000;
      const windowSize = 120; // 2 minutes
      const windowStart = now - windowSize;
      const features = computeSlidingWindowFeatures(actionHistory, windowStart, now);
      if (features) {
        setInteractionFeatures(features);
        // Update ML prediction on every window update
        updatePredictionFromML();
      }
    }, 100); // Update every 0.1 second
    return () => clearInterval(interval);
  }, [actionHistory]);

  // --- Track interaction by pushing to action history ---
  const trackInteraction = useCallback((interactionType: string, details?: any) => {
    const now = Date.now() / 1000;
    setActionHistory(prev => [
      ...prev,
      { timestamp: now, action: interactionType, details: details || {} }
    ]);
  }, []);

  // --- Listen for scroll events and push to action history ---
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

  // --- Types for Sliding Window Feature Extraction ---
  interface ActionDetails {
    scroll_amount?: number;
  }

  interface UserAction {
    timestamp: number;
    action: string;
    details?: ActionDetails;
  }

  interface SlidingWindowFeatures {
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
  }

  // --- Sliding Window Feature Extraction (matches Python logic) ---
  function computeSlidingWindowFeatures(
    actionHistory: UserAction[],
    windowStart: number,
    windowEnd: number
  ): SlidingWindowFeatures | null {
    // Clamp window to not exceed current time
    const timeNow = Date.now() / 1000;
    // Filter actions in this window
    const windowActions = actionHistory.filter((a: UserAction) => a.timestamp >= windowStart && a.timestamp <= windowEnd && a.timestamp <= timeNow);
    if (windowActions.length < 5) return null; // Not enough data

    // Temporal features
    const timestamps = windowActions.map((a: UserAction) => a.timestamp);
    const timeSinceLastAction =
      timestamps.length > 0
        ? Math.max(0, windowEnd - timestamps[timestamps.length - 1])
        : windowEnd - windowStart;
    let avgTimeBetweenActions: number;
    if (timestamps.length > 1) {
      const intervals = timestamps.slice(1).map((t, i) => t - timestamps[i]);
      avgTimeBetweenActions = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    } else {
      avgTimeBetweenActions = windowEnd - windowStart;
    }
    const sessionDuration = windowEnd - (actionHistory.length > 0 ? actionHistory[0].timestamp : windowEnd);

    // Count actions (normalized by window size)
    const actionCounts: Record<string, number> = {
      scroll_down: 0, scroll_up: 0, click: 0, hover: 0, blur_window: 0, focus_window: 0, wait: 0, close_tab: 0
    };
    windowActions.forEach((a: UserAction) => { if (actionCounts[a.action] !== undefined) actionCounts[a.action] += 1; });
    const windowDuration = windowEnd - windowStart;
    const scrollDownCount = actionCounts.scroll_down / windowDuration;
    const scrollUpCount = actionCounts.scroll_up / windowDuration;
    const clickCount = actionCounts.click / windowDuration;
    const hoverCount = actionCounts.hover / windowDuration;
    const blurCount = actionCounts.blur_window / windowDuration;
    const focusCount = actionCounts.focus_window / windowDuration;
    const waitCount = actionCounts.wait / windowDuration;
    const closeTabCount = actionCounts.close_tab / windowDuration;

    // Scroll depth and velocity
    let scrollDepth = 0;
    let scrollAmounts: number[] = [];
    windowActions.forEach((a: UserAction) => {
      if (a.action === 'scroll_down' || a.action === 'scroll_up') {
        const amount = a.details?.scroll_amount ?? 0.1;
        if (a.action === 'scroll_down') scrollDepth += amount;
        else scrollDepth -= amount;
        scrollAmounts.push(amount);
      }
    });
    const scrollDepthMax = Math.max(0, scrollDepth);
    const scrollVelocityAvg = scrollAmounts.length > 0 ? scrollAmounts.reduce((a, b) => a + b, 0) / scrollAmounts.length : 0;

    // Interaction density
    const interactionDensity = windowActions.length / windowDuration;

    // Attention score
    const focusBalance = focusCount - blurCount;
    const attentionScore = Math.max(0, Math.min(1, 0.5 + focusBalance * 10 + interactionDensity * 0.1));

    // Action entropy
    const totalActions = Object.values(actionCounts).reduce((a, b) => a + b, 0);
    let actionEntropy = 0;
    if (totalActions > 0) {
      const probs = Object.values(actionCounts).map(c => c / totalActions);
      actionEntropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p + 1e-10) : 0), 0);
    }

    // Burstiness score
    let burstinessScore = 0;
    if (timestamps.length > 2) {
      const intervals = timestamps.slice(1).map((t, i) => t - timestamps[i]);
      const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const std = Math.sqrt(intervals.map(x => (x - mean) ** 2).reduce((a, b) => a + b, 0) / intervals.length);
      burstinessScore = std / (mean + 1e-10);
    }

    // Engagement rhythm
    let engagementRhythm = 0;
    if (timestamps.length > 3) {
      const intervals = timestamps.slice(1).map((t, i) => t - timestamps[i]);
      if (intervals.length > 1) {
        // Simple autocorrelation
        const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        const num = intervals.slice(0, -1).map((v, i) => (v - mean) * (intervals[i + 1] - mean)).reduce((a, b) => a + b, 0);
        const den = intervals.map(x => (x - mean) ** 2).reduce((a, b) => a + b, 0);
        engagementRhythm = den > 0 ? Math.max(0, num / den) : 0;
      }
    }

    return {
      time_since_last_action: timeSinceLastAction,
      avg_time_between_actions: avgTimeBetweenActions,
      session_duration: sessionDuration,
      scroll_down_count: scrollDownCount,
      scroll_up_count: scrollUpCount,
      scroll_depth_max: scrollDepthMax,
      interaction_density: interactionDensity,
      attention_score: attentionScore,
      scroll_velocity_avg: scrollVelocityAvg,
      action_entropy: actionEntropy,
      burstiness_score: burstinessScore,
      engagement_rhythm: engagementRhythm
    };
  }
  // --- End Sliding Window Feature Extraction ---

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