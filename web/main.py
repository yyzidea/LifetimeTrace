from flask import Flask, render_template, request, jsonify
import numpy as np
import time

import sys
sys.path.append('/Users/mac/Documents/Python/LifetimeTrace')
try:
    from lifetime_trace import LifetimeTrace
except AssertionError:
    pass

# meas: LifetimeTrace
app = Flask(__name__)


def status_json(status, description):
    return jsonify({'status': status, 'description': description})


@app.route("/")
def hello_world():
    return render_template('index.html', ver=time.strftime('%Y%m%d%H%M%S', time.localtime()))


@app.route("/get_test_data")
def get_test_data():
    return jsonify(np.random.random(6).tolist())


@app.route("/measurement")
def measurement():
    global meas

    command = request.args.get('command')
    if command == 'create':
        try:
            meas = LifetimeTrace(request.args.get('click_channels'), request.args.get('start_channels'),
                                 request.args.get('binwidth'), request.args.get('n_bins'),
                                 request.args.get('int_time'), serial=request.args.get('serial'))
            return status_json(1, 'Tagger is created.')
        except Exception as inst:
            print(inst)
            return status_json(0, 'Some errors happen during creating tagger.')
    elif meas in dir() and meas is not None:
        if command == 'start':
            meas.startFor(request.args.get('duration'))
            return status_json(1, 'Measurement starts.')
        elif command == 'stop':
            meas.stop()
            return status_json(1, 'Measurement stops.')
        elif command == 'get_data':
            return jsonify(meas.getData())
        elif command == 'get_new_data':
            return jsonify(meas.getData())
        else:
            return status_json(0, 'Illegal measurement command.')
    else:
        return status_json(0, 'Tagger is None.')


# @app.route("/test")
# def test():
#     global t
#     print(type(t))


@app.route("/api")
def api():
    if request.args.get('script') == 'chart':
        return app.send_static_file('js/chart.js')
    elif request.args.get('script') == 'app':
        return app.send_static_file('js/app.js')


if __name__ == '__main__':
    # app.run(host, port, debug, options)
    # 默认值：host=127.0.0.1, port=5000, debug=false
    app.run(debug=True)
