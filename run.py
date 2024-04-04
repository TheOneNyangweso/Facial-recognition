import subprocess


def run_streamlit_app(app_name):
    """
    Run the Streamlit app.
    """
    streamlit_command = f'streamlit run {app_name}'
    subprocess.Popen(["gnome-terminal", "--", "bash", "-c", streamlit_command])

if __name__ == "__main__":
    run_streamlit_app("trial.py")