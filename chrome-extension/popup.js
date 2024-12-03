document.getElementById('fillForms').addEventListener('click', () => {
    // Query the active tab
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length > 0) {
            const activeTabId = tabs[0].id; // Get the active tab ID

            // Execute the script on the active tab
            chrome.scripting.executeScript({
                target: { tabId: activeTabId },
                function: fillForm // Ensure the function is defined
            }, () => {
                if (chrome.runtime.lastError) {
                    console.error(chrome.runtime.lastError.message);
                }
            });
        } else {
            console.error("No active tab found.");
        }
    });
});

// Define the function to be injected into the active tab
// function fillForm() {
//     // Perform form-filling logic in the context of the active tab
//     fetch('http://localhost:5055/api/auto_fill', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({ html_content: document.documentElement.innerHTML })
//     })
//     .then(response => response.json())
//     .then(data => {
//         for (const [id, value] of Object.entries(data)) {
//             const field = document.getElementById(id);
//             if (field) field.value = value;
//         }
//     })
//     .catch(error => console.error('Error auto-filling form:', error));
// }
// function fillForm() {
//     const htmlContent = document.documentElement.innerHTML;
//     const requestBody = {
//         url: window.location.href, // Add the current URL to the request
//         html_content: htmlContent
//     };

//     console.log('Request body:', requestBody); // Log the request body for debugging

//     fetch('http://localhost:5055/api/auto_fill', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify(requestBody)
//     })
//     .then(response => {
//         if (!response.ok) {
//             throw new Error(`Server responded with status ${response.status}`);
//         }
//         return response.json();
//     })
//     .then(data => {
//         console.log('Received response:', data); // Log the response from the API

//         for (const [id, value] of Object.entries(data)) {
//             const field = document.getElementById(id);
//             if (field) {
//                 field.value = value;
//                 console.log(`Filled field ${id} with value: ${value}`);
//             } else {
//                 console.warn(`Field with ID ${id} not found`);
//             }
//         }
//     })
//     .catch(error => console.error('Error auto-filling form:', error));
// }

function fillForm() {
    const htmlContent = document.documentElement.innerHTML;
    const requestBody = {
        url: window.location.href,
        html_content: htmlContent
    };

    fetch('http://localhost:5055/api/auto_fill', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Received response:', data); // Log the response from the API

        for (const [key, value] of Object.entries(data)) {
            let field = document.getElementById(key); // Try finding by ID first
            if (!field) {
                // If not found by ID, try finding by placeholder or label text
                field = Array.from(document.querySelectorAll('input, select, textarea'))
                    .find(el => el.getAttribute('placeholder') === key || el.getAttribute('aria-label') === key);
            }

            if (field) {
                field.value = value;
                console.log(`Filled field ${key} with value: ${value}`);
            } else {
                console.warn(`Field with key ${key} not found`);
            }
        }
    })
    .catch(error => console.error('Error auto-filling form:', error));
}
