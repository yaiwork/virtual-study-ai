import React, { useState } from 'react';

const BACKEND_URL = 'https://virtual-study-ai.onrender.com';

export default function Chat() {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState([]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    const newHistory = [...history, userMsg];
    setHistory(newHistory);
    setInput('');

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input })
      });
      const data = await res.json();
      setHistory(h => [...h, { role: 'assistant', content: data.answer }]);
    } catch (err) {
      setHistory(h => [...h, { role: 'assistant', content: 'Error: ' + err }]);
    }
  };

  return (
    <div>
      <div className="chat-box">
        {history.map((msg, idx) => (
          <div key={idx} className={`msg ${msg.role}`}>{msg.content}</div>
        ))}
      </div>
      <textarea value={input} onChange={e => setInput(e.target.value)} rows={3} />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}
