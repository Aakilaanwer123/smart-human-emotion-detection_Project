# Smart Human Emotion Detection — Web Version

This is a Streamlit web adaptation of the Tkinter application. It provides:
- Webcam snapshot and uploaded-photo emotion analysis
- Browser microphone recording and uploaded-audio analysis
- Face/voice result comparison
- Supportive emotion companion text
- Emotion challenge quiz
- SQLite history, dashboard and CSV export

## Run locally

1. Install Python 3.10 or 3.11.
2. Open this folder in VS Code.
3. Create and activate a virtual environment:
   - Windows PowerShell: `python -m venv venv`
   - `.env\Scripts\Activate.ps1`
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Run: `streamlit run app.py`

The first voice-model use downloads model files and requires internet. Webcam/microphone access is requested by the browser.

## Deploy as a website

1. Push `app.py`, `requirements.txt`, and this README to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose the repository, branch, and `app.py`, then deploy.
4. Share the resulting Streamlit URL with your team.

## Important deployment notes

- This is a web adaptation, not a direct conversion of Tkinter. Some interface elements differ.
- Face/voice predictions are estimates, not diagnoses or proof of someone's inner feelings.
- Streamlit Community Cloud may struggle with large ML dependencies or model download time. If deployment fails due to memory/build limits, use a server with sufficient memory or separate model inference into an API.
- SQLite on a hosted instance is not a reliable shared permanent database. Use a hosted database if team members must share persistent history.
- Do not commit private photos, voice recordings, credentials, or database files to a public repository.
