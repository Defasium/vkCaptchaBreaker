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

// This block is new!
chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    chrome.storage.sync.get('power', function(data) {
		const current = data.power;
		const message = current ? 'power_on' : 'power_off';
		sendResponse({data:'responded'});
		chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
			chrome.tabs.sendMessage(tabs[0].id, {"message": message});
		});
	});
  }
);


chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
	if (changeInfo.status === 'complete') {
		chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {			
			const url_array = tabs[0].url.split('/');
			const suburl = url_array[url_array.length-1];
			if (suburl.startsWith('mail') || suburl.startsWith('write')) {
				chrome.tabs.sendMessage(tabs[0].id, {"message": "relocate_target"});
			} else {
				chrome.tabs.sendMessage(tabs[0].id, {"message": "disable_observer"});
			}
		});
	}
});
