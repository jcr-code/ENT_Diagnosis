from flask import Flask, render_template, request, redirect, session, url_for
import json, sys
import urllib.request, secrets , requests 
from train import *

app = Flask(__name__)
secret_key = secrets.token_hex(16)
app.secret_key = secret_key


#====================================================================================
# Home Page
#====================================================================================
@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('form.html')

#====================================================================================
# Access Data
#====================================================================================
@app.route('/submit-form', methods=['POST'])
def submit_form():
    # Extract form data
    form_data = request.form
    # Pass form data to function in another Python file
    # Kelas Data Hadling
    outputs = handle_form_data(form_data)
    # Redirect to another route or URL
    # return redirect(url_for('index'))
    return render_template('results.html', outputs=outputs)
    
if __name__ == "__main__":
    app.run(debug=True)