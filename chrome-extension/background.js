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
});
