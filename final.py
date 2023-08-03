import http.server
import socketserver
import urllib.parse
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("dataset.csv", sep=",", index_col=0)
data = df.copy()

data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.set_index("timestamp")

# Normalizing data
# Min-max normalization
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Isolation Forest
contamination = 0.001
data3 = data.copy()
iso_forest = IsolationForest(contamination=contamination, random_state=42)
iso_forest.fit_predict(data3)


PORT = 8000

class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        query_params = dict(urllib.parse.parse_qs(parsed_url.query))
        values_array = list(query_params.values())
        columns = ['ram_value_percentage', 'cpu_usage_percent', 'io_usage_percent']
        for i in range(len(values_array)):
            values_array[i][0] = float(values_array[i][0])
        
        df = pd.DataFrame()
        for col in columns:
          df[col] = values_array[i]

        df = scaler.transform(df)

        print('df for predict is\n', df)

        if parsed_url.path == '/iso':
          res = iso_forest.predict(df)          
          response_body = []
          if(res==-1):
            response_body.append("It is an anomaly\n")
          else:
            response_body.append("It is not an anomaly\n")

          response_body = response_body[0].encode()
                
          self.send_response(200)
          self.send_header('Content-type', 'application/json')
          self.end_headers()
          self.wfile.write(response_body)

        else:
          super().do_GET()


with socketserver.TCPServer(("", PORT), MyRequestHandler) as httpd:
    print("Server is running on port", PORT)
    httpd.serve_forever()
