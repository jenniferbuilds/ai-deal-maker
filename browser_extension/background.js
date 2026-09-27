// Background service worker

chrome.runtime.onInstalled.addListener(() => {
    console.log('AI Deal Maker extension installed');
    
    // Set default settings
    chrome.storage.local.set({
        apiEndpoint: 'http://localhost:8080',
        autoMode: false,
        requireApproval: true
    });
});

// Handle messages between content script and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'callAPI') {
        // Make API call to local deal maker service
        fetch(request.endpoint, {
            method: request.method || 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(request.data)
        })
        .then(response => response.json())
        .then(data => sendResponse({success: true, data: data}))
        .catch(error => sendResponse({success: false, error: error.toString()}));
        
        return true; // Keep channel open for async response
    }
});
