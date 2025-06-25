import React, { useState } from 'react';

const BACKEND_URL = 'https://virtual-study-ai.onrender.com';

export default function HomeworkAssistant() {
  const [question, setQuestion] = useState('');
  const [file, setFile] = useState(null);
  const [answer, setAnswer] = useState('');

  const submit = async () => {
    if (!question && !file) {
      setAnswer('Please provide a question or upload a file.');
      return;
    }

    try {
      let response;
      if (file) {
        const formData = new FormData();
        if (question) formData.append('question', question);
        formData.append('file', file);
        response = await fetch(`${BACKEND_URL}/homework-assistant`, {
          method: 'POST',
          body: formData
        });
      } else {
        response = await fetch(`${BACKEND_URL}/homework-assistant`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
      }
      const data = await response.json();
      setAnswer(data.answer || 'No response');
    } catch (err) {
      setAnswer('Error: ' + err);
    }
  };

  return (
    <div>
      <textarea
        placeholder="Enter your homework question"
        value={question}
        onChange={e => setQuestion(e.target.value)}
        rows={3}
      />
      <input type="file" onChange={e => setFile(e.target.files[0])} />
      <button onClick={submit}>Get Help</button>
      {answer && <div className="result" dangerouslySetInnerHTML={{__html: answer}} />}
    </div>
  );
}
