import { useEffect, useRef, useCallback } from 'react';
import { usePrivAds } from '../components/PrivAdsProvider';

export const useInteractionTracker = () => {
  const { trackInteraction } = usePrivAds();
  const lastMouseMove = useRef<number>(0);
  const scrollTimeout = useRef<NodeJS.Timeout | null>(null);
  const timeSpentInterval = useRef<NodeJS.Timeout | null>(null);
  const startTime = useRef<number>(Date.now());
  const lastInteractionTime = useRef<number>(0);

  // Throttle interactions to prevent excessive updates
  const throttledTrackInteraction = useCallback((type: string, details?: any) => {
    const now = Date.now();
    const timeSinceLastInteraction = now - lastInteractionTime.current;

    // Only allow interactions every 200ms minimum
    if (timeSinceLastInteraction < 200) {
      return;
    }

    lastInteractionTime.current = now;
    trackInteraction(type, details);
  }, [trackInteraction]);

  useEffect(() => {
    // Track mouse movements (throttled)
    const handleMouseMove = (e: MouseEvent) => {
      const now = Date.now();
      if (now - lastMouseMove.current > 500) { // Throttle to every 500ms for mouse moves
        throttledTrackInteraction('mouse_move', {
          x: e.clientX,
          y: e.clientY,
          timestamp: now
        });
        lastMouseMove.current = now;
      }
    };

    // Track scrolling
    const handleScroll = () => {
      if (scrollTimeout.current) {
        clearTimeout(scrollTimeout.current);
      }

      scrollTimeout.current = setTimeout(() => {
        const scrollPercent = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
        throttledTrackInteraction('scroll', {
          scrollPercent: Math.round(scrollPercent),
          scrollY: window.scrollY
        });
      }, 300); // Debounce scroll events
    };

    // Track clicks
    const handleClick = (e: MouseEvent) => {
      throttledTrackInteraction('click', {
        target: (e.target as HTMLElement)?.tagName || 'unknown',
        x: e.clientX,
        y: e.clientY
      });
    };

    // Track hovers on interactive elements
    const handleMouseEnter = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target && target.closest && (target.tagName === 'BUTTON' || target.tagName === 'A' || target.closest('button') || target.closest('a'))) {
        throttledTrackInteraction('hover', {
          target: target.tagName,
          element: target.textContent?.slice(0, 20) || 'unknown'
        });
      }
    };

    // Track time spent
    timeSpentInterval.current = setInterval(() => {
      const secondsSpent = Math.floor((Date.now() - startTime.current) / 1000);
      if (secondsSpent > 0 && secondsSpent % 10 === 0) { // Every 10 seconds
        throttledTrackInteraction('time_spent', { seconds: secondsSpent });
      }
    }, 1000);

    // Add event listeners
    document.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('scroll', handleScroll);
    document.addEventListener('click', handleClick);
    document.addEventListener('mouseenter', handleMouseEnter, true); // Use capture phase

    // Cleanup
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('scroll', handleScroll);
      document.removeEventListener('click', handleClick);
      document.removeEventListener('mouseenter', handleMouseEnter, true);

      if (scrollTimeout.current) {
        clearTimeout(scrollTimeout.current);
      }
      if (timeSpentInterval.current) {
        clearInterval(timeSpentInterval.current);
      }
    };
  }, [throttledTrackInteraction]);

  // Return current interaction metrics for debugging
  return {
    timeSpent: Math.floor((Date.now() - startTime.current) / 1000),
    lastActivity: lastMouseMove.current
  };
};