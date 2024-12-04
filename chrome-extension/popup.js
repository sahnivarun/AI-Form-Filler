document.getElementById('fillForms').addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length > 0) {
            const activeTabId = tabs[0].id;

            // Inject the content script into the active tab
            chrome.scripting.executeScript(
                {
                    target: { tabId: activeTabId },
                    files: ['content.js'], // Ensure the content script is injected
                },
                () => {
                    if (chrome.runtime.lastError) {
                        console.error('Error injecting script:', chrome.runtime.lastError.message);
                    } else {
                        console.log('Content script executed successfully.');
                    }
                }
            );
        } else {
            console.error('No active tab found.');
        }
    });
});
