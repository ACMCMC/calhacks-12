// File: privads_demo/src/components/CustomizedAd.tsx
import React, { useEffect, useState } from 'react';
import './CustomizedAd.css';

interface PageContext {
  title: string;
  content: string;
  keywords: string[];
  page_type: string;
  url: string;
  summary_text: string;
}

interface CustomizedAdProps {
  pageContext?: PageContext;
}

const CustomizedAd: React.FC<CustomizedAdProps> = ({ pageContext: providedContext }) => {
  const [adText, setAdText] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [adId, setAdId] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);

  // Use provided context or create a default
  const pageContext = providedContext || {
    title: 'Article',
    content: '',
    keywords: [],
    page_type: 'news',
    url: typeof window !== 'undefined' ? window.location.href : '',
    summary_text: '',
  };

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    const fetchAd = async () => {
      setLoading(true);
      setError(null);

      try {
        console.log('[CustomizedAd] 1️⃣ Fetching ad with context:', pageContext);

        const response = await fetch('http://localhost:8002/web_ad/from_context', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            page_context: pageContext,
            user_embedding: null,
            user_id: null,
            injection_method: 'web_component',
          }),
          signal: controller.signal,
        });

        if (!isMounted) {
          console.log('[CustomizedAd] Component unmounted, ignoring response');
          return;
        }

        console.log('[CustomizedAd] 2️⃣ Got response status:', response.status);

        if (!response.ok) {
          throw new Error(`API Error: ${response.status}`);
        }

        console.log('[CustomizedAd] 3️⃣ Parsing JSON...');
        const data = await response.json();
        console.log('[CustomizedAd] 4️⃣ Response data:', data);

        if (!isMounted) return;

        if (data.success) {
          console.log('[CustomizedAd] 5️⃣ Success! Setting ad text:', data.customized_ad_text);
          setAdText(data.customized_ad_text);
          setAdId(data.ad_id);
          setConfidence(data.confidence_score);
          console.log('[CustomizedAd] 6️⃣ ✅ Ad state updated!');
        } else {
          console.log('[CustomizedAd] ❌ API returned success=false:', data.error_message);
          setError(data.error_message || 'Failed to fetch ad');
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          console.log('[CustomizedAd] Request was cancelled');
          return;
        }
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        console.error('[CustomizedAd] ❌ Error caught:', errorMessage);
        console.error('[CustomizedAd] Full error:', err);
        if (isMounted) {
          setError(errorMessage);
        }
      } finally {
        if (isMounted) {
          console.log('[CustomizedAd] 7️⃣ Setting loading=false');
          setLoading(false);
        }
      }
    };

    fetchAd();

    return () => {
      console.log('[CustomizedAd] Cleanup: marking unmounted and aborting request');
      isMounted = false;
      controller.abort();
    };
  }, [JSON.stringify(pageContext)]);

  if (loading) {
    return (
      <div className="customized-ad-container">
        <div className="ad-loading">⏳ Loading personalized ad...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="customized-ad-container">
        <div className="ad-error">
          ❌ Error loading ad: {error}
        </div>
      </div>
    );
  }

  if (!adText) {
    return null;
  }

  return (
    <div className="customized-ad-container">
      <div className="ad-header">
        <span className="ad-badge">🎯 SPONSORED - PRIVADS</span>
      </div>
      <div className="ad-body">
        <p className="ad-text">{adText}</p>
        <button className="ad-button">Learn More</button>
      </div>
      <div className="ad-footer">
        <span className="ad-meta">Ad ID: {adId}</span>
        <span className="ad-confidence">Confidence: {confidence?.toFixed(2)}</span>
      </div>
    </div>
  );
};

export default CustomizedAd;