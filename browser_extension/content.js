// Content script - runs on marketplace pages

let negotiationActive = false;
let itemInfo = {};
let negotiationHistory = [];

// Platform-specific selectors
const PLATFORMS = {
    'facebook.com': {
        name: 'Facebook Marketplace',
        titleSelector: '[data-testid="marketplace_item_title"]',
        priceSelector: '[data-testid="marketplace_item_price"]',
        messageInputSelector: '[data-testid="message_input"]',
        sendButtonSelector: '[data-testid="send_button"]'
    },
    'craigslist.org': {
        name: 'Craigslist',
        titleSelector: '.postingtitle',
        priceSelector: '.price',
        messageInputSelector: '#reply-to-email',
        sendButtonSelector: 'button[type="submit"]'
    },
    'offerup.com': {
        name: 'OfferUp',
        titleSelector: '[data-testid="itemTitle"]',
        priceSelector: '[data-testid="itemPrice"]',
        messageInputSelector: '[data-testid="messageInput"]',
        sendButtonSelector: '[data-testid="sendButton"]'
    }
};

// Detect which platform we're on
function detectPlatform() {
    const hostname = window.location.hostname;
    for (const [domain, config] of Object.entries(PLATFORMS)) {
        if (hostname.includes(domain)) {
            return config;
        }
    }
    return null;
}

// Extract item info from page
function extractItemInfo(platformConfig) {
    const titleEl = document.querySelector(platformConfig.titleSelector);
    const priceEl = document.querySelector(platformConfig.priceSelector);
    
    let price = 0;
    if (priceEl) {
        const priceText = priceEl.textContent.replace(/[^0-9.]/g, '');
        price = parseFloat(priceText) || 0;
    }
    
    return {
        title: titleEl ? titleEl.textContent.trim() : 'Unknown Item',
        price: price,
        url: window.location.href,
        platform: platformConfig.name
    };
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const platformConfig = detectPlatform();
    
    if (request.action === 'getItemInfo') {
        if (platformConfig) {
            itemInfo = extractItemInfo(platformConfig);
            sendResponse(itemInfo);
        } else {
            sendResponse({error: 'Platform not supported'});
        }
    }
    
    else if (request.action === 'startNegotiation') {
        negotiationActive = true;
        const settings = request.settings;
        
        // Start monitoring for seller messages
        monitorConversation(platformConfig, settings);
        
        sendResponse({success: true});
    }
    
    else if (request.action === 'stopNegotiation') {
        negotiationActive = false;
        sendResponse({success: true});
    }
    
    else if (request.action === 'suggestMessage') {
        // Get AI suggestion
        getSuggestion(request.maxPrice, request.targetPrice)
            .then(suggestion => sendResponse({suggestion: suggestion}));
        return true; // Keep channel open for async response
    }
    
    return true;
});

// Monitor conversation for new messages
function monitorConversation(platformConfig, settings) {
    const observer = new MutationObserver((mutations) => {
        if (!negotiationActive) return;
        
        // Check for new seller messages
        // Platform-specific logic needed here
        console.log('Monitoring for new messages...');
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}

// Get AI suggestion for next message
async function getSuggestion(maxPrice, targetPrice) {
    try {
        // Call local API server
        const response = await fetch('http://localhost:8080/suggest', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                item: itemInfo,
                maxPrice: maxPrice,
                targetPrice: targetPrice,
                history: negotiationHistory
            })
        });
        
        const data = await response.json();
        return data.suggestion;
    } catch (error) {
        console.error('Error getting suggestion:', error);
        return "Error: Could not connect to AI service. Is the deal maker service running?";
    }
}

// Send message in chat
function sendMessage(message, platformConfig) {
    const inputEl = document.querySelector(platformConfig.messageInputSelector);
    const sendBtn = document.querySelector(platformConfig.sendButtonSelector);
    
    if (inputEl && sendBtn) {
        // Set message text
        inputEl.value = message;
        inputEl.dispatchEvent(new Event('input', {bubbles: true}));
        
        // Click send
        setTimeout(() => {
            sendBtn.click();
            
            // Add to history
            negotiationHistory.push({
                role: 'buyer',
                message: message,
                timestamp: Date.now()
            });
        }, 100);
    }
}

console.log('AI Deal Maker content script loaded');
