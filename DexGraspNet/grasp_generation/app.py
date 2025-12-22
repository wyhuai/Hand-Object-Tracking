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

@app.route('/data/experiments_usb_005/exp_32/results/<int:id>')
def serve_html(id):
    # Serve the HTML file based on the ID provided
    filename = f"sem-USBStick-1baa93373407c8924315bea999e66ce3_index_{id}.html"  # Adjust this based on your naming convention
    return send_from_directory('/data/experiments_usb_005/exp_32/results', filename)

if __name__ == '__main__':
    app.run(port=5000)  # Run your Flask app