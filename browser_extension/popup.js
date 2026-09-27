// Popup script for AI Deal Maker extension

document.addEventListener('DOMContentLoaded', function() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const suggestBtn = document.getElementById('suggest-btn');
    const maxPriceInput = document.getElementById('max-price');
    const targetPriceInput = document.getElementById('target-price');
    const autoModeCheckbox = document.getElementById('auto-mode');
    const statusDiv = document.getElementById('status');
    
    // Load saved settings
    chrome.storage.local.get(['maxPrice', 'targetPrice', 'autoMode'], function(result) {
        if (result.maxPrice) maxPriceInput.value = result.maxPrice;
        if (result.targetPrice) targetPriceInput.value = result.targetPrice;
        if (result.autoMode !== undefined) autoModeCheckbox.checked = result.autoMode;
    });
    
    // Get current tab info
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        const tab = tabs[0];
        
        // Send message to content script to get item info
        chrome.tabs.sendMessage(tab.id, {action: 'getItemInfo'}, function(response) {
            if (response && response.title) {
                document.getElementById('item-title').textContent = response.title;
                document.getElementById('item-price').textContent = '$' + response.price;
                
                // Suggest max price (70% of asking)
                maxPriceInput.value = Math.round(response.price * 0.7);
                targetPriceInput.value = Math.round(response.price * 0.6);
            }
        });
    });
    
    // Start negotiation
    startBtn.addEventListener('click', function() {
        const settings = {
            maxPrice: parseFloat(maxPriceInput.value),
            targetPrice: parseFloat(targetPriceInput.value) || parseFloat(maxPriceInput.value) * 0.85,
            autoMode: autoModeCheckbox.checked
        };
        
        // Save settings
        chrome.storage.local.set(settings);
        
        // Start negotiation in content script
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            chrome.tabs.sendMessage(tabs[0].id, {
                action: 'startNegotiation',
                settings: settings
            }, function(response) {
                if (response && response.success) {
                    statusDiv.textContent = '🤖 Negotiation active';
                    statusDiv.style.background = '#d4edda';
                }
            });
        });
    });
    
    // Stop negotiation
    stopBtn.addEventListener('click', function() {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            chrome.tabs.sendMessage(tabs[0].id, {action: 'stopNegotiation'}, function(response) {
                if (response && response.success) {
                    statusDiv.textContent = 'Stopped';
                    statusDiv.style.background = '#f0f0f0';
                }
            });
        });
    });
    
    // Suggest next message
    suggestBtn.addEventListener('click', function() {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            chrome.tabs.sendMessage(tabs[0].id, {
                action: 'suggestMessage',
                maxPrice: parseFloat(maxPriceInput.value),
                targetPrice: parseFloat(targetPriceInput.value)
            }, function(response) {
                if (response && response.suggestion) {
                    // Show suggestion in a prompt for user to copy
                    prompt('Suggested response:', response.suggestion);
                }
            });
        });
    });
});
