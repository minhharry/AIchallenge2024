import os

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

PATH = "E:\\Downloads\\atm24949_withsound\\keyframes"
list_img = []
for path in os.listdir(PATH):
    for filename in os.listdir(os.path.join(PATH, path)):
        list_img.append("images/" + path + '/' + filename)

@app.route('/')
def index():
    return jsonify(list_img)

@app.route('/images/<path:path>/<filename>')
def serve_image(path, filename):
    return send_from_directory(os.path.join(PATH, path), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) 