from flask import Flask, render_template
import subprocess
import sys
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/run_script',  methods=['POST'])

def runscript():
    # Path to the Python interpreter in your virtual environment
    python_path = sys.executable
    script_path = 'emotion.py'
    subprocess.Popen([python_path, script_path])
    return 'Script is running in the background'

if __name__ == '__main__':
    app.run(debug=True)