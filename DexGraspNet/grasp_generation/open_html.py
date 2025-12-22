import webbrowser
import time



# Loop through the specified range of indices
for index in range(0, 127):
    # Construct the URL
    url = f'http://localhost:5000/results/{index}'
    # Open the URL in the default web browser
    webbrowser.open(url)
