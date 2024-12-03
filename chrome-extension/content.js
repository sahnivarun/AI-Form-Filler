fetch('http://localhost:5055/api/auto_fill', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ html_content: document.documentElement.innerHTML })
})
.then(response => response.json())
.then(data => {
    for (const [id, value] of Object.entries(data)) {
        const field = document.getElementById(id);
        if (field) field.value = value;
    }
})
.catch(error => console.error('Error auto-filling form:', error));
