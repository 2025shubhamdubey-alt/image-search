import subprocess
import uvicorn
import threading

def run_streamlit():
    subprocess.Popen(["streamlit", "run", "Streamlit/streamlit_ui.py", "--server.port", "8501"])

if __name__ == "__main__":
    threading.Thread(target=run_streamlit).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8000)