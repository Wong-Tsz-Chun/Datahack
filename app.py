from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html', active_section='about')

@app.route('/services')
def services():
    return render_template('index.html', active_section='services')

@app.route('/contact')
def contact():
    return render_template('index.html', active_section='contact')

# New route for the chat page
@app.route('/chat')
def chat():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)