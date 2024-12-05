chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "fetch-auto-fill") {
        const apiUrl = "http://localhost:5055/api/auto_fill";

        // Make the API call to fetch form data
        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ html_content: message.html_content }),
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`Server responded with status ${response.status}`);
                }
                return response.json();
            })
            .then((data) => {
                // Send the response back to the content script
                sendResponse({ success: true, data });
            })
            .catch((error) => {
                console.error('Error fetching auto-fill data:', error);
                sendResponse({ success: false, error: error.message });
            });

        return true; // Keep the message channel open for asynchronous response
    }
    
    else if (message.action === "generate-cover-letter") {
        const jobUrl = message.url;

        // Fetch the job description content from the URL
        fetch(jobUrl)
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch job description. Status: ${response.status}`);
                }
                return response.text();
            })
            .then((htmlContent) => {
                // Send the job description content to the LLM API
                const apiUrl = "http://localhost:5055/api/generate_cover_letter";

                fetch(apiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ job_description: htmlContent }),
                })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error(`Server responded with status ${response.status}`);
                    }
                    return response.json();
                })
                .then((data) => {
                    // Send the generated cover letter back to popup.js
                    sendResponse({ success: true, coverLetter: data.cover_letter });
                })
                .catch((error) => {
                    console.error('Error generating cover letter:', error);
                    sendResponse({ success: false, error: error.message });
                });
            })
            .catch((error) => {
                console.error('Error fetching job description:', error);
                sendResponse({ success: false, error: error.message });
            });

        return true; // Keep the message channel open for asynchronous response
    }
});
