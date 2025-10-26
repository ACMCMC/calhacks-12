// Content script for PrivAds extension
(function() {
  'use strict';
  
  let isExtensionActive = false;
  let injectedAds = new Set();
  
  // Initialize extension
  init();
  
  function init() {
    // Check if extension should be active on this page
    checkExtensionStatus();
    
    // Listen for messages from background script
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === 'injectAdCode') {
        injectAdCode(request.injectionCode, request.adId);
        sendResponse({ success: true });
      }
    });
    
    // Add UI controls
    addExtensionControls();
  }
  
  function checkExtensionStatus() {
    // Check if user has enabled ads for this domain
    chrome.storage.local.get(['enabledDomains', 'extensionEnabled'], (result) => {
      const enabledDomains = result.enabledDomains || [];
      const extensionEnabled = result.extensionEnabled !== false; // Default to true
      const currentDomain = window.location.hostname;
      
      isExtensionActive = extensionEnabled && (
        enabledDomains.length === 0 || // No restrictions
        enabledDomains.includes(currentDomain) // Domain is enabled
      );
      
      updateUI();
    });
  }
  
  function addExtensionControls() {
    // Create floating control panel
    const controlPanel = document.createElement('div');
    controlPanel.id = 'privads-control-panel';
    controlPanel.innerHTML = `
      <div class="privads-controls">
        <button id="privads-toggle" class="privads-btn">Enable Ads</button>
        <button id="privads-quick-test" class="privads-btn">Quick Test</button>
        <button id="privads-settings" class="privads-btn">Settings</button>
      </div>
    `;
    
    document.body.appendChild(controlPanel);
    
    // Add event listeners
    document.getElementById('privads-toggle').addEventListener('click', toggleExtension);
    document.getElementById('privads-quick-test').addEventListener('click', quickTest);
    document.getElementById('privads-settings').addEventListener('click', openSettings);
  }
  
  function toggleExtension() {
    isExtensionActive = !isExtensionActive;
    
    chrome.storage.local.set({ extensionEnabled: isExtensionActive });
    updateUI();
    
    if (isExtensionActive) {
      // Trigger ad injection for current page
      injectAdForCurrentPage();
    } else {
      // Remove all injected ads
      removeAllAds();
    }
  }
  
  function quickTest() {
    if (!isExtensionActive) {
      alert('Please enable ads first');
      return;
    }
    
    const testAdText = 'This is a test advertisement injected by PrivAds extension!';
    
    chrome.runtime.sendMessage({
      action: 'quickAd',
      url: window.location.href,
      adText: testAdText
    }, (response) => {
      if (response.success) {
        console.log('Quick test ad injected successfully');
      } else {
        console.error('Quick test ad failed:', response.error);
      }
    });
  }
  
  function openSettings() {
    // Open extension popup or settings page
    chrome.runtime.sendMessage({ action: 'openSettings' });
  }
  
  function injectAdForCurrentPage() {
    if (!isExtensionActive) return;
    
    // Get user embedding from storage or generate default
    chrome.storage.local.get(['userEmbedding'], (result) => {
      const userEmbedding = result.userEmbedding || generateDefaultEmbedding();
      
      chrome.runtime.sendMessage({
        action: 'injectAd',
        url: window.location.href,
        userEmbedding: userEmbedding
      }, (response) => {
        if (response.success) {
          console.log('Ad injected successfully:', response.adId);
        } else {
          console.error('Ad injection failed:', response.error);
        }
      });
    });
  }
  
  function injectAdCode(injectionCode, adId) {
    if (!isExtensionActive) return;
    
    try {
      // Create a temporary container
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = injectionCode;
      
      // Extract the main ad container
      const adContainer = tempDiv.firstElementChild;
      if (adContainer) {
        // Add to page
        document.body.appendChild(adContainer);
        
        // Track injected ad
        injectedAds.add(adId);
        
        // Add removal button
        addRemovalButton(adContainer, adId);
        
        console.log(`Ad ${adId} injected successfully`);
      }
      
    } catch (error) {
      console.error('Error injecting ad code:', error);
    }
  }
  
  function addRemovalButton(adContainer, adId) {
    const removeBtn = document.createElement('button');
    removeBtn.innerHTML = '×';
    removeBtn.className = 'privads-remove-btn';
    removeBtn.title = 'Remove this ad';
    
    removeBtn.addEventListener('click', () => {
      adContainer.remove();
      injectedAds.delete(adId);
    });
    
    adContainer.appendChild(removeBtn);
  }
  
  function removeAllAds() {
    const adContainers = document.querySelectorAll('[id^="privads-container-"]');
    adContainers.forEach(container => container.remove());
    injectedAds.clear();
  }
  
  function generateDefaultEmbedding() {
    // Generate a random 128-dimensional embedding
    return Array.from({ length: 128 }, () => Math.random() - 0.5);
  }
  
  function updateUI() {
    const toggleBtn = document.getElementById('privads-toggle');
    if (toggleBtn) {
      toggleBtn.textContent = isExtensionActive ? 'Disable Ads' : 'Enable Ads';
      toggleBtn.className = `privads-btn ${isExtensionActive ? 'active' : ''}`;
    }
  }
  
  // Auto-inject ads when page loads (if extension is active)
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      setTimeout(injectAdForCurrentPage, 2000); // Wait 2 seconds after page load
    });
  } else {
    setTimeout(injectAdForCurrentPage, 2000);
  }
  
  // Listen for page changes (for SPAs)
  let lastUrl = location.href;
  new MutationObserver(() => {
    const url = location.href;
    if (url !== lastUrl) {
      lastUrl = url;
      setTimeout(injectAdForCurrentPage, 1000); // Inject after navigation
    }
  }).observe(document, { subtree: true, childList: true });
  
})();

