// document.getElementById('fillForms').addEventListener('click', () => {
//     chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
//         if (tabs.length > 0) {
//             const activeTabId = tabs[0].id;

//             // Inject the content script into the active tab
//             chrome.scripting.executeScript(
//                 {
//                     target: { tabId: activeTabId },
//                     files: ['content.js'], // Ensure the content script is injected
//                 },
//                 () => {
//                     if (chrome.runtime.lastError) {
//                         console.error('Error injecting script:', chrome.runtime.lastError.message);
//                     } else {
//                         console.log('Content script executed successfully.');
//                     }
//                 }
//             );
//         } else {
//             console.error('No active tab found.');
//         }
//     });
// });

//Event Listener for Form Fill
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

// Event listener for the "Generate Cover Letter" button
document.getElementById('generateCoverLetter').addEventListener('click', () => {
    const jobUrl = document.getElementById('jobUrl').value;

    if (jobUrl) {
        // Send message to background script to generate cover letter
        chrome.runtime.sendMessage(
            { action: 'generate-cover-letter', url: jobUrl },
            (response) => {
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
