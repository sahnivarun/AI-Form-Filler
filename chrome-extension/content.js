async function attachResumeFromURL(field, url) {
    console.log("Fetching the resume file from URL:", url);
    
    const driveFileIdMatch = url.match(/file\/d\/([^\/]+)/);
    if (driveFileIdMatch) {
        const fileId = driveFileIdMatch[1];
        url = `https://drive.google.com/uc?export=download&id=${fileId}`;
        console.log("Converted Google Drive URL to direct download link:", url);
    }
    
    try {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Failed to fetch resume file from url. Status: ${response.status}`);
        }

        const blob = await response.blob();
        const fileName = "resume.pdf"; // Provide a default filename if none exists
        const file = new File([blob], fileName, { type: blob.type });

        if (!(file instanceof File)) {
            throw new TypeError("Fetched file is not a valid File object.");
        }

        console.log("Fetched file from URL:", file);

        // Create a DataTransfer object to simulate user file selection
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);

        // Attach the file to the input field
        field.files = dataTransfer.files;
        console.log("File attached from URL:", field.files[0]);

        console.log("File attached from URL:", field.files[0]);
        field.dispatchEvent(new Event('change', { bubbles: true, cancelable: true }));
        console.log("Change event dispatched for file from URL.");
    } catch (error) {
        console.error('Error attaching resume from URL:', error);
    }
}

async function attachResume(field, localFile, url) {
    console.log("Attaching resume to field:", field);
    console.log("Local File:", localFile, "URL:", url);

    try {
        if (localFile && localFile.content) {
            console.log("Reconstructing local file from serialized data...");
            const byteString = atob(localFile.content.split(',')[1]); // Decode Base64
            const arrayBuffer = new ArrayBuffer(byteString.length);
            const uint8Array = new Uint8Array(arrayBuffer);

            for (let i = 0; i < byteString.length; i++) {
                uint8Array[i] = byteString.charCodeAt(i);
            }

            const reconstructedFile = new File([uint8Array], localFile.name, { type: localFile.type });
            console.log("Reconstructed File:", reconstructedFile);

            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(reconstructedFile);
            field.files = dataTransfer.files;

            console.log("Local file attached:", field.files[0]);
            field.dispatchEvent(new Event('change', { bubbles: true, cancelable: true }));
            console.log("Change event dispatched for local file.");
        } else if (url) {
            console.log("Fetching and attaching file from URL...");
            await attachResumeFromURL(field, url);
        } else {
            console.error("No resume file or URL provided.");
        }
    } catch (error) {
        console.error("Error attaching resume:", error);
    }
}


async function fetchAndFillForm(resumeURL, localFile = null) {
    const htmlContent = document.documentElement.innerHTML;

    console.log("Starting form fill process...");
    console.log("Resume URL:", resumeURL);
    console.log("Local File:", localFile);

    // Send a message to the background script to fetch auto-fill data
    chrome.runtime.sendMessage(
        { action: "fetch-auto-fill", html_content: htmlContent },
        async (response) => {
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

                        // Handle file inputs (resume attachment)
                        else if (field.type === 'file') {
                            console.log(`Handling file input for field: ${key}`);
                            try {
                                await attachResume(field, localFile, resumeURL);
                                console.log(`Resume successfully attached to field: ${key}`);
                            } catch (error) {
                                console.error(`Error attaching resume to field: ${key}`, error);
                            }
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

                // Notify the popup script that form filling is complete
                console.log("Form filling completed successfully. Sending success message to popup.");
                chrome.runtime.sendMessage({ action: "form-fill-complete" });
            } 
            
            else {
                console.error('Error filling form:', response.error);

                // Notify the popup script about the error
                console.log("Sending error message to popup.");
                chrome.runtime.sendMessage({ action: "form-fill-error", error: response.error });
            }
        }
    );
}


// Listen for messages from popup.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'fill-form') {
        console.log("Message received for form fill:", message);
        fetchAndFillForm(message.resumeURL, message.resumeFile);
        sendResponse({ success: true });
    }
    return true;
});
