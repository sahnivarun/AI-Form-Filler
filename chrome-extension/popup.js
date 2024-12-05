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
    const resumeInput = document.getElementById('resumeInput').files[0]; // Local resume file
    const resumeURL = document.getElementById('resumeURL').value.trim(); // Resume URL

    console.log("Resume Input:", resumeInput);
    if (resumeInput) {
        console.log("Resume Input Type:", resumeInput.constructor.name);
    }

    console.log("Resume URL:", resumeURL);

    if (!resumeInput && !resumeURL) {
        alert('Please provide a resume file or a URL.');
        return;
    }

    // If no local file but URL is provided, fetch the resume first
    if (!resumeInput && resumeURL) {
        showLoader();
        chrome.runtime.sendMessage({ action: 'fetch-resume', url: resumeURL }, (response) => {
            hideLoader();
            if (response && response.success) {
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
                                    console.error('Error injecting content script:', chrome.runtime.lastError.message);
                                    alert('Error filling the form: ' + chrome.runtime.lastError.message);
                                    hideLoader(); // Hide loader on error
                                } 
                                else {
                                    console.log('Content script injected successfully.');

                                    chrome.tabs.sendMessage(activeTabId, { 
                                        action: 'fill-form', 
                                        resumeURL, 
                                        resumeFile: {
                                            name: "my_resume.pdf",
                                            type: "application/pdf",
                                            size: 0,
                                            content: response.fileContent
                                        }
                                    }, (msgResponse) => {
                                        if (chrome.runtime.lastError) {
                                            console.error("Error sending message to content script:", chrome.runtime.lastError.message);
                                        } else {
                                            console.log("Message sent to content script successfully. Response:", msgResponse);
                                        }
                                    });
                                }
                            }
                        );
                
                        // Listen for messages from the content script
                        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
                            console.log("Message received from content script:", message);
                            
                            if (message.action === 'form-fill-complete') {
                                console.log("Form filling action completed successfully. Details:", message);
                                hideLoader(); // Hide loader once the form-filling is done
                                alert('Form filled successfully!');
                                sendResponse({ success: true });
                            } 
                            
                            else if (message.action === 'form-fill-error') {
                                console.error('Error during form filling:', message.error);
                                hideLoader();
                                alert('Error filling the form: ' + message.error);
                                sendResponse({ success: false });
                            }

                            else {
                                console.warn("Unexpected message action:", message.action);
                            }
                            
                            return true; // Keeps the listener open for async response
                        });
                    } 
                    
                    else {
                        console.error('No active tab found.');
                        alert('No active tab found.');
                        hideLoader(); // Hide loader if no active tab is found
                    }
                });
            } else {
                console.error("Error fetching resume from URL:", response ? response.error : 'Unknown error');
                alert('Error fetching resume from URL: ' + (response ? response.error : 'Unknown error'));
            }
        });
        return;
    }

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
                        console.error('Error injecting content script:', chrome.runtime.lastError.message);
                        alert('Error filling the form: ' + chrome.runtime.lastError.message);
                        hideLoader(); // Hide loader on error
                    } 
                    
                    else {
                        console.log('Content script injected successfully.');

                        // If a local file is selected, serialize it for passing
                        if (resumeInput) {
                            const reader = new FileReader();
                            reader.onload = () => {
                                chrome.tabs.sendMessage(activeTabId, { 
                                    action: 'fill-form', 
                                    resumeURL, 
                                    resumeFile: {
                                        name: resumeInput.name,
                                        type: resumeInput.type,
                                        size: resumeInput.size,
                                        content: reader.result, // Serialized content (Base64)
                                    }
                                }, (response) => {
                                    if (chrome.runtime.lastError) {
                                        console.error("Error sending message to content script:", chrome.runtime.lastError.message);
                                    } else {
                                        console.log("Message sent to content script successfully. Response:", response);
                                    }
                                });
                            };
                            reader.readAsDataURL(resumeInput); // Read file as Base64
                        } else {
                            // Send only the URL if no local file is provided
                            chrome.tabs.sendMessage(activeTabId, { 
                                action: 'fill-form', 
                                resumeURL, 
                                resumeFile: null
                            }, (response) => {
                                if (chrome.runtime.lastError) {
                                    console.error("Error sending message to content script:", chrome.runtime.lastError.message);
                                } else {
                                    console.log("Message sent to content script successfully. Response:", response);
                                }
                            });
                        }
                    }
                }
            );
    

            // Listen for messages from the content script
            chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
                
                console.log("Message received from content script:", message);
                
                if (message.action === 'form-fill-complete') {
                    console.log("Form filling action completed successfully. Details:", message);
                    hideLoader(); // Hide loader once the form-filling is done
                    alert('Form filled successfully!');
                    sendResponse({ success: true });
                } 
                
                else if (message.action === 'form-fill-error') {
                    console.error('Error during form filling:', message.error);
                    hideLoader();
                    alert('Error filling the form: ' + message.error);
                    sendResponse({ success: false });
                }

                else {
                    console.warn("Unexpected message action:", message.action);
                }
                
                return true; // Keeps the listener open for async response
            });
        } 
        
        else {
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
                } 
                
                else {
                    console.error('Error generating cover letter:', response.error);
                    alert('Error generating cover letter: ' + response.error);
                }
            }
        );
    } 
    
    else {
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
