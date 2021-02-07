// background.js

var prior = {
	states: [{message: 'power_off'}, {message: 'power_on'}],
	icons: [{path: 'icon48_0.png'}, {path: 'icon48_1.png'}],
	powers: [{power: 0}, {power: 1}],
}

chrome.runtime.onInstalled.addListener(function() {
  chrome.storage.sync.set(prior.powers[0]);
});

async function updateIcon(meta) {	
  chrome.storage.sync.get('power', function(data) {
    const current = data.power ^ 1;
    chrome.browserAction.setIcon(prior.icons[current]);
    chrome.storage.sync.set(prior.powers[current], function() {
      console.log('The number is set to ' + current);
    });
	sendMessage(prior.states[current], meta.id);
  });
};

function sendMessage(message, id) {
	if (!id) {
		chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
			chrome.tabs.sendMessage(tabs[0].id, message);
		});
	} else {
		chrome.tabs.sendMessage(id, message);
	}
}

// Called when the user clicks on the browser action.
chrome.browserAction.onClicked.addListener(updateIcon);

async function messageCallback(request, sender, sendResponse) {
    if (typeof request.data === 'string') {
        chrome.storage.sync.get('power', function(data) {
            sendMessage(prior.states[data.power], sender.tab.id);
        });
	}
};
// This block is new!
chrome.runtime.onMessage.addListener(messageCallback);
