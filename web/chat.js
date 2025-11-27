const chatBox = document.getElementById('chat-box');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function addMessage(text, sender) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'msg ' + sender;
  msgDiv.textContent = text;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.onclick = async () => {
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, 'user');
  input.value = '';
  sendBtn.disabled = true;

  // Call backend (adjust the URL if needed)
  try {
    const res = await fetch('http://localhost:8000/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: text})
    });
    const data = await res.json();
    addMessage(data.answer, 'bot');
  } catch (error) {
    addMessage('Sorry, connection error.', 'bot');
  }
  sendBtn.disabled = false;
};

input.addEventListener('keyup', e => {
  if (e.key === 'Enter') sendBtn.click();
});
