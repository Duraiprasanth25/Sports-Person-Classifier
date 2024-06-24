from flask import Flask, request, jsonify
import util
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/classify_image', methods = ['GET', 'POST'])
def classify_image():

        image_data = request.form['image_data']


        response = jsonify(util.classify_image(image_base64_data=image_data))
        response.headers.add('Access_Control_Allow_Origin','*')

        return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000,debug=True)

