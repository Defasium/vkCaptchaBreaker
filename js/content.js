// content.js
const data = {
	width: 128,
	height: 64,
	available: null,
	flag: null,
	config: {
	  subtree: true,
	  attributes: false,
	  childList: true,
	  characterData: false
	},
	global_observer: new MutationObserver(manageObserver),
	oc: document.createElement('canvas'),
	captcha: {
		aInternal: '',
		aListener: function(val) {},
		set a(val) {
			this.aInterval = val;
			this.aListener(val);
		},
		get(a) {
			return this.aInterval;
		},
		registerListener: function(listener) {
			this.aListener = listener;
		}
	},
	RGBimgarr: new Uint8Array(128*64*3),
};

async function wait_image(click=true) {
	const img = document.getElementsByClassName('captcha_img')[0];
	if (img == null) return true;
	if (!img.complete) {
        await (new Promise(r => {img.onload = r})).then();
	}
	// Image have loaded.
	return await recognize_captcha(img, click);
}

async function recognize_captcha(img, click) {
	const placeholder = document.getElementsByName('captcha_key')[0];
	var bool_recognized = true;
	const octx = data.oc.getContext('2d');
	octx.drawImage(img, 0, 0, data.width, data.height);
	// Run model with Tensor inputs in background and get the result.
	data.RGBimgarr.fill(0);
	const arr = octx.getImageData(0, 0, data.width, data.height).data;
	for (var i=0, counter=0, length=data.width*data.height*4; i<length; i++) {
		if ((i+1) % 4){
			data.RGBimgarr[counter] += arr[i];
			counter++;
		}
	}
	chrome.runtime.sendMessage(data.RGBimgarr);
	await (new Promise(r => {data.captcha.registerListener(r)})
		).then((captcha) => {
		if (captcha==='unreachable'){
			bool_recognized = false;
		} else {
			placeholder.value = captcha;
		}
	});
	if (!bool_recognized)
		return bool_recognized;
	if (click) {
		const img_src = img.src+'';
		var submit_button = document.getElementsByClassName('mailDialog__confirmButton');
		if (submit_button.length === 0) {
			// if captcha is on own page then there is no need in double checking:
			// page will be redirected anyway
			document.getElementsByClassName('wide_button')[0].click();
			return true;
		}
		submit_button[0].click();
        try {
			if (document.getElementsByClassName('captcha_img')[0])
				if (document.getElementsByClassName('captcha_img')[0].src !== img_src)
					bool_recognized = false;
		} catch (e) {
			console.log(e);
			bool_recognized = true;
		}
		return bool_recognized;
	}
	return true;
}

chrome.runtime.sendMessage({data: '?'});

async function iconCallback(request, sender, sendResponse) {
	if (request.captcha) {
		data.captcha.a = request.captcha;
		return;
	}
	console.log(request);
	if (request.message === 'power_on') {
		data.available = true;
		data.flag = true;
		data.oc.width = data.width;
		data.oc.height = data.height;
		manageObserver(null, data.global_observer);
    } else if(request.message === 'power_off') {
		data.available = true;
		data.flag = true;
	}
	data.global_observer.observe(document.body, data.config);
}

chrome.runtime.onMessage.addListener(iconCallback);

async function manageObserver(mutations, obs) {
	if (document.getElementsByClassName('captcha_img').length === 0) {
		data.flag = true;
		return;
	}
	if (data.flag && data.available) {
		obs.disconnect();
		const click = document.getElementsByName('pass')[0] ? false : true;
		try{
			data.available = false;
			var i = 0;
			while (i++<5) {
				try {
					if (await wait_image(click)) break;
				} catch (e) {
					console.log(e);
					break;
				}
			}
			if (i>=5)
				chrome.runtime.sendMessage({data: '?'});
		} catch (e) {console.log(e);}
		data.available = true;
		obs.observe(document.body, data.config);
		data.flag = false;
	}
}