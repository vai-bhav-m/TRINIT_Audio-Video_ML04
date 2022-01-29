import flask
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/',methods = ['GET'])
def show_index_html():
    return render_template('index.html')

@app.route('/send_data', methods = ['POST'])
def get_data_from_html():
        input_date = request.form['time-start']
        print(f'Date is {input_date}')
        return "Data sent. Please check your program log"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)