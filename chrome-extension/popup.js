// Show the loader
function showLoader() {
    document.getElementById('loader').style.display = 'flex';
}

// Hide the loader
function hideLoader() {
    document.getElementById('loader').style.display = 'none';
}

// Event Listener for Form Fill
document.getElementById('fillForms').addEventListener('click', () => {
    showLoader(); // Show loader while filling the form

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length > 0) {
            const activeTabId = tabs[0].id;

            // Inject the content script into the active tab
            chrome.scripting.executeScript(
                {
                    target: { tabId: activeTabId },
                    files: ['content.js'],
                },
                (injectionResults) => {
                    if (chrome.runtime.lastError) {
                        console.error('Error injecting script:', chrome.runtime.lastError.message);
                        alert('Error filling the form: ' + chrome.runtime.lastError.message);
                        hideLoader(); // Hide loader on error
                    } else {
                        console.log('Content script executed successfully.');
                    }
                }
            );

            // Listen for messages from the content script
            chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
                if (message.action === 'form-fill-complete') {
                    console.log('Form filling completed successfully.');
                    hideLoader(); // Hide loader once the form-filling is done
                    alert('Form filled successfully!');
                    sendResponse({ success: true });
                } else if (message.action === 'form-fill-error') {
                    console.error('Error during form filling:', message.error);
                    hideLoader(); // Hide loader on error
                    alert('Error filling the form: ' + message.error);
                    sendResponse({ success: false });
                }
                return true; // Keeps the listener open for async response
            });
        } else {
            console.error('No active tab found.');
            alert('No active tab found.');
            hideLoader(); // Hide loader if no active tab is found
        }
    });
});

// Event Listener for "Generate Cover Letter" button
document.getElementById('generateCoverLetter').addEventListener('click', () => {
    const jobUrl = document.getElementById('jobUrl').value;

    if (jobUrl) {
        showLoader(); // Show loader while generating the cover letter
        chrome.runtime.sendMessage(
            { action: 'generate-cover-letter', url: jobUrl },
            (response) => {
                hideLoader(); // Hide loader once the process is complete
                if (response && response.success) {
                    const coverLetter = response.coverLetter;
                    document.getElementById('coverLetter').value = coverLetter;
                } else {
                    console.error('Error generating cover letter:', response.error);
                    alert('Error generating cover letter: ' + response.error);
                }
            }
        );
    } else {
        alert('Please enter a job description URL.');
    }
});

// Event Listener for "Copy" button
document.getElementById('copyCoverLetter').addEventListener('click', () => {
    const coverLetter = document.getElementById('coverLetter');
    if (coverLetter.value.trim() === '') {
        alert('The cover letter content is empty. Generate the cover letter first.');
        return;
    }

    coverLetter.select();
    coverLetter.setSelectionRange(0, 99999);
    document.execCommand('copy');
    alert('Cover letter copied to clipboard!');
});
