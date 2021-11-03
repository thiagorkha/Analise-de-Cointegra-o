import os
from flask import Flask, render_template
app1 = Flask(__name__)

@app1.route('/')
def index():
  return render_template("index.html")

app1.run(host='0.0.0.0', port=8501)
os.system("streamlit run coint.py")

  