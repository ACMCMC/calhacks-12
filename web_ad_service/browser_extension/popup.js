// Popup script for PrivAds extension
document.addEventListener('DOMContentLoaded', function() {
  const mainToggle = document.getElementById('mainToggle');
  const statusMessage = document.getElementById('statusMessage');
  const quickTestBtn = document.getElementById('quickTestBtn');
  const injectCurrentBtn = document.getElementById('injectCurrentBtn');
  const delayInput = document.getElementById('delayInput');
  const methodSelect = document.getElementById('methodSelect');
  
  // Load saved settings
  loadSettings();
  
  // Event listeners
  mainToggle.addEventListener('click', toggleExtension);
  quickTestBtn.addEventListener('click', quickTest);
  injectCurrentBtn.addEventListener('click', injectCurrentPage);
  delayInput.addEventListener('change', saveSettings);
  methodSelect.addEventListener('change', saveSettings);
  
  function loadSettings() {
    chrome.storage.local.get(['extensionEnabled', 'injectionDelay', 'injectionMethod'], (result) => {
      const isEnabled = result.extensionEnabled !== false; // Default to true
      mainToggle.classList.toggle('active', isEnabled);
      
      delayInput.value = result.injectionDelay || 2000;
      methodSelect.value = result.injectionMethod || 'web_component';
    });
  }
  
  function saveSettings() {
    const settings = {
      injectionDelay: parseInt(delayInput.value),
      injectionMethod: methodSelect.value
    };
    
    chrome.storage.local.set(settings);
    showStatus('Settings saved', 'success');
  }
  
  function toggleExtension() {
    chrome.storage.local.get(['extensionEnabled'], (result) => {
      const newState = !result.extensionEnabled;
      
      chrome.storage.local.set({ extensionEnabled: newState }, () => {
        mainToggle.classList.toggle('active', newState);
        showStatus(newState ? 'Extension enabled' : 'Extension disabled', 'success');
        
        // Notify content script
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          if (tabs[0]) {
            chrome.tabs.sendMessage(tabs[0].id, {
              action: 'toggleExtension',
              enabled: newState
            });
          }
        });
      });
    });
  }
  
  function quickTest() {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'quickTest'
        }, (response) => {
          if (chrome.runtime.lastError) {
            showStatus('Error: ' + chrome.runtime.lastError.message, 'error');
          } else if (response && response.success) {
            showStatus('Quick test ad injected!', 'success');
          } else {
            showStatus('Quick test failed', 'error');
          }
        });
      }
    });
  }
  
  function injectCurrentPage() {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'injectCurrentPage'
        }, (response) => {
          if (chrome.runtime.lastError) {
            showStatus('Error: ' + chrome.runtime.lastError.message, 'error');
          } else if (response && response.success) {
            showStatus('Ad injected for current page!', 'success');
          } else {
            showStatus('Injection failed', 'error');
          }
        });
      }
    });
  }
  
  function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = `status ${type}`;
    statusMessage.style.display = 'block';
    
    setTimeout(() => {
      statusMessage.style.display = 'none';
    }, 3000);
  }
  
  // Check API connection
  checkAPIConnection();
  
  function checkAPIConnection() {
    fetch('http://localhost:8002/web_ad/health')
      .then(response => response.json())
      .then(data => {
        if (data.status === 'healthy') {
          showStatus('API connected', 'success');
        } else {
          showStatus('API connection issue', 'error');
        }
      })
      .catch(error => {
        showStatus('API not available', 'error');
        console.error('API connection error:', error);
      });
  }
});

