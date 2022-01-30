import flask
from flask import Flask, render_template, request
app = Flask(__name__)
import datetime
import arima_model

@app.route('/',methods = ['GET'])
def show_index_html():
    return render_template('index.html')

@app.route('/send_data', methods = ['POST'])
def get_data_from_html():
        input_date = request.form['time-start']
        current_date = datetime.datetime.strptime(input_date,"%Y-%m-%d")
        current_date += datetime.timedelta(days=1)
        next_date = current_date.strftime("%Y-%m-%d")

        prediction, mae = arima_model.Final_model_predict(input_date)
        return render_template('index.html', today=input_date, tomorrow=next_date, prediction=prediction, error=mae)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)