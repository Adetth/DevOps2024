from  flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/welcome')
def welcome():
    return 'Welcome to ci cd with git hub actions and ec2!'

@app.route('/add/<int:num1>/<int:num2>')
def add(num1, num2):
    return str(num1 + num2)

@app.route('/sub/<int:num1>/<int:num2>')
def sub(num1, num2):
    return str(num1 - num2)



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)