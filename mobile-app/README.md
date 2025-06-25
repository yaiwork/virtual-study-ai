# Virtual Study AI Mobile

This folder contains a simple React front-end that mirrors the features of the Streamlit UI. It communicates with the FastAPI backend provided in `backend/app.py`.

## Available Scripts

- `npm start` – start the development server
- `npm run build` – create a production build

The app is mobile friendly and provides five modes:

1. **Chat** – ask any question
2. **Homework Assistant** – upload files or ask questions about homework
3. **Lesson Plan Generator** – generate lesson plans based on subject, grade and topic
4. **Exam Questions Generator** – create exam questions
5. **Study Guide Generator** – generate study guides

Make sure the backend is running (by default at `https://virtual-study-ai.onrender.com`) or update `BACKEND_URL` constants inside the components.
