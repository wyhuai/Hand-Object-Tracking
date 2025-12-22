from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/save_id', methods=['POST'])
def save_id():
    data = request.get_json()
    id = data.get('id')

    # Save the ID to a text file
    with open('saved_ids.txt', 'a') as f:
        f.write(f"{id}\n")

    return jsonify({'message': 'ID saved successfully'}), 200

@app.route('/results/<int:id>')
def serve_html(id):
    # Serve the HTML file based on the ID provided
    filename = f"ddg-kit_CokePlasticLarge_index_{id}.html"  # Adjust this based on your naming convention
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(port=5000)  # Run your Flask app