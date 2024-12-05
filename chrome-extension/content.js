// Function to fetch and fill form
function fetchAndFillForm() {
    const htmlContent = document.documentElement.innerHTML;

    // Send a message to the background script to fetch auto-fill data
    chrome.runtime.sendMessage(
        { action: "fetch-auto-fill", html_content: htmlContent },
        (response) => {
            if (response.success) {
                const data = response.data;
                console.log('Received form data:', data);

                // Fill the form with the received data
                for (const [key, fieldData] of Object.entries(data)) {
                    let field = document.getElementById(key); // Try finding by ID first
                    if (!field) {
                        // Fallback to finding by name
                        field = document.querySelector(`[name="${key}"]`);
                    }
                    if (!field) {
                        // Further fallback using placeholder or aria-label
                        field = Array.from(document.querySelectorAll('input, select, textarea')).find(
                            (el) =>
                                el.getAttribute('placeholder') === key ||
                                el.getAttribute('aria-label') === key
                        );
                    }

                    if (field) {
                        // Handle radio buttons
                        if (field.type === 'radio') {
                            const radioGroup = document.querySelectorAll(`[name="${field.name}"]`);
                            radioGroup.forEach((radio) => {
                                if (radio.value === fieldData) {
                                    radio.checked = true;

                                    // Trigger input and change events for validation/UI updates
                                    radio.dispatchEvent(new Event('input', { bubbles: true }));
                                    radio.dispatchEvent(new Event('change', { bubbles: true }));
                                    console.log(`Filled radio group ${key} with value: ${fieldData}`);
                                }
                            });
                        }
                        // Handle dropdown/select elements
                        else if (field.tagName === 'SELECT') {
                            field.value = fieldData;

                            // Trigger change event for validation/UI updates
                            field.dispatchEvent(new Event('change', { bubbles: true }));
                            console.log(`Filled select field ${key} with value: ${fieldData}`);
                        }
                        // Handle other input types (text, textarea, etc.)
                        else {
                            field.value = fieldData;

                            // Trigger input and change events for validation/UI updates
                            field.dispatchEvent(new Event('input', { bubbles: true }));
                            field.dispatchEvent(new Event('change', { bubbles: true }));
                            console.log(`Filled field ${key} with value: ${fieldData}`);
                        }
                    } else {
                        console.warn(`Field with key ${key} not found.`);
                    }
                }
            } else {
                console.error('Error filling form:', response.error);
            }
        }
    );
}

// Immediately run the fetchAndFillForm function
fetchAndFillForm();
