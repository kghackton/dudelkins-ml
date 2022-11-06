import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from sys import argv
import logging
import torch
import json
import time

class NeuralNetwork():
    def __init__(self):
        self.model = torch.load('./weights/best_878.pt', map_location=torch.device('cpu'))
        self.model.to('cpu').eval()

        with open('./_id_defect_dict.json', 'r') as jsonfile1:
            self.id_defect_dict = json.load(jsonfile1)

        with open('./_id_done_works_dict.json', 'r') as jsonfile2:
            self.id_done_works_dict = json.load(jsonfile2)

        with open('./_id_done_security_works_dict.json', 'r') as jsonfile3:
            self.id_done_security_works = json.load(jsonfile3)

        self.label_map = {'resolved': 0, 'consulted': 1, 'reject': 2, 'cancel': 3}

    def predict(self, message):
        with torch.no_grad():
            id_defect = float(self.id_defect_dict[message['id_defect']])
            id_emergency = 0.0 if message['id_emergency'] == 'normal' else 1.0
            done_works = [float(self.id_done_works_dict[x]) for x in message['id_done_works']]
            if message['id_security_works'] == []:
                security_works = [0]
            else:
                security_works = [float(self.id_done_security_works[x]) for x in message['id_security_works']]
            label = self.label_map[message['result']]
            # print(id_defect, id_emergency, done_works, security_works, label)

            batch_input = []
            batch_label = []
            for dw in done_works:
                for ddw in security_works:
                    sample = torch.tensor((id_defect / 335.0, id_emergency, dw / 1256.0, ddw / 38.0), dtype=torch.float)
                    batch_input.append(sample)
                    labels = torch.tensor(label, dtype=torch.long)
                    batch_label.append(labels)

                    # print(id_defect, id_emergency, dw, ddw, label)
            batch_input = torch.stack(batch_input, dim=0)
            batch_label = torch.stack(batch_label, dim=0)

            output = self.model(batch_input)

            acc = (output.argmax(1) == batch_label).sum().item()
            _label = int(batch_label[0])
            confidence = float(output[:, _label].mean())
            # print(values)
            if acc > 0:
                return (False, confidence)
            return (True, 1.0 - confidence)
    def dummy(self):
        return None


class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        model = NeuralNetwork()
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), post_data.decode('utf-8'))

        body = post_data.decode('utf-8')
        message = json.loads(body)
        t2 = time.time()
        abnormal, confidence = model.predict(message)
        t3 = time.time()
        logging.info(f"Time for inference: {t3-t2}")
        logging.info(f"Answer of Neural Network: {abnormal} with confidence: {confidence}")

        # make some dirt with Neural Network

        self._set_response()
        if abnormal:
            dict_for_send = json.dumps({"isAbnormal": True, "confidence": confidence}).encode('utf-8')
            self.wfile.write(dict_for_send)
        else:
            dict_for_send = json.dumps({"isAbnormal": False, "confidence": confidence}).encode('utf-8')
            self.wfile.write(dict_for_send)


def run(server_class=HTTPServer, handler_class=S, port=42069):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    # active_neural_network = handler_class.activate_nn('1')
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()