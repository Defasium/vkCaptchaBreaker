// background.js

chrome.runtime.onInstalled.addListener(function() {
  chrome.storage.sync.set({power: 0}, function() {
    console.log('Is inactive');
  });
});

function updateIcon() {	
  chrome.storage.sync.get('power', function(data) {
    var current = data.power;
	if (++current > 1) current = 0;
    chrome.browserAction.setIcon({'path': 'icon48_' + current + '.png'});
    chrome.storage.sync.set({power: current}, function() {
		console.log('The number is set to ' + current);
    });
	chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
		var message = 'power_on';
		if (current === 0) message = 'power_off';
		chrome.tabs.sendMessage(tabs[0].id, {"message": message});
	});
  });
};

// Called when the user clicks on the browser action.
chrome.browserAction.onClicked.addListener(function(tab) {
  // Send a message to the active tab
  updateIcon();
});
