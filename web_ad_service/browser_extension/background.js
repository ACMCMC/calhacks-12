// Background service worker for PrivAds extension
chrome.runtime.onInstalled.addListener(() => {
  console.log('PrivAds extension installed');
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'injectAd') {
    handleAdInjection(request, sender, sendResponse);
    return true; // Keep message channel open for async response
  }
});

async function handleAdInjection(request, sender, sendResponse) {
  try {
    const { url, userEmbedding, injectionMethod = 'web_component' } = request;
    
    // Call the PrivAds API
    const response = await fetch('http://localhost:8002/web_ad/complete', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url: url,
        user_embedding: userEmbedding,
        injection_method: injectionMethod,
        position: 'bottom'
      })
    });
    
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.success) {
      // Send injection code to content script
      chrome.tabs.sendMessage(sender.tab.id, {
        action: 'injectAdCode',
        injectionCode: data.injection_code,
        adId: data.ad_id
      });
      
      sendResponse({ success: true, adId: data.ad_id });
    } else {
      sendResponse({ success: false, error: data.error_message });
    }
    
  } catch (error) {
    console.error('Error in ad injection:', error);
    sendResponse({ success: false, error: error.message });
  }
}

// Handle quick ad injection for testing
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'quickAd') {
    handleQuickAd(request, sender, sendResponse);
    return true;
  }
});

async function handleQuickAd(request, sender, sendResponse) {
  try {
    const { url, adText } = request;
    
    const response = await fetch('http://localhost:8002/web_ad/quick', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url: url,
        ad_text: adText || 'Check out this amazing product!'
      })
    });
    
    const data = await response.json();
    
    if (data.success) {
      chrome.tabs.sendMessage(sender.tab.id, {
        action: 'injectAdCode',
        injectionCode: data.injection_code,
        adId: 'quick_test'
      });
      
      sendResponse({ success: true });
    } else {
      sendResponse({ success: false, error: 'Quick ad generation failed' });
    }
    
  } catch (error) {
    console.error('Error in quick ad:', error);
    sendResponse({ success: false, error: error.message });
  }
}

