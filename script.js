document.getElementById('chat-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    const question = document.getElementById('question').value;
    console.log('Question submitted:', question);
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        const result = await response.json();
        document.getElementById('answer').innerText = result.answer;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('answer').innerText = 'Error: ' + error.message;
    }
});
