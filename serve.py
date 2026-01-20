import flask
import megfile
import io
import json
import math
import av
import cv2
import numpy as np
#import av
import time
import decord
import boto3
import redis
import json

rd = redis.StrictRedis('localhost')
s3_client = boto3.client('s3', endpoint_url = 'http://tos-s3-cn-beijing.ivolces.com')

app = flask.Flask(__name__)

@app.route('/setlabel/<int:frameid>/<path:path>', methods = ['POST'])
def set_label(frameid, path):
    label = flask.request.get_json(force = True)
    rd.hset('label_result_' + path, frameid, json.dumps(label))
    print(path, frameid, label)
    return flask.jsonify({'success': True})

@app.route('/label/<robot_name>/<int:taskid>')
def label_frame(robot_name, taskid):
    task_list = json.loads(rd.get('label_task_list_' + robot_name))
    task = task_list[taskid]
    path = task['path'] + '/videos/cam_high_rgb.mp4'
    frameid = task['frameid']
    labeled = rd.hget('label_result_' + path, frameid)
    print(path, frameid, labeled)
    if labeled is None:
        labeled = {}
    else:
        labeled = json.loads(labeled.decode('utf8'))
    info = {
        'robot_name' : robot_name,
        'taskid' : taskid,
        'frameid' : frameid,
        'path' : path,
        'labeled' : labeled
    }
    return flask.render_template('label.html', info = info)

def read_frame(path, idx):
    alt_path = path.replace('s3://', 's3://unsullied/sharefs/fhq/')
    try:
        bucket, key = alt_path[5:].split('/', 1)
        s3_client.head_object(Bucket=bucket, Key=key)
        path = alt_path
    except s3_client.exceptions.ClientError:
        pass
    with megfile.smart_open(path, mode='rb', block_size = 32768) as fin:
        container = av.open(fin)
        stream = container.streams.video[0]
        #print(stream.average_rate, stream.time_base, idx)
        target_pts = int(idx / stream.time_base / stream.average_rate)
        #print('target_pts', target_pts)
        container.seek(target_pts, stream=stream, backward = True)
        for _, frame in enumerate(container.decode(stream)):
            #print('pts', frame.pts)
            if frame.pts >= target_pts:
                break
        if path.endswith('.mkv'):
            frame = frame.to_ndarray()
            frame = np.float32(frame) / max(1, frame.max())
            frame = np.uint8(frame * 255 * 8)
        else:
            frame = frame.to_rgb().to_ndarray()[:, :, ::-1].copy()
    return frame

@app.route('/image/<int:idx>/<path:path>')
def get_frame(idx, path):
    # get target size from query string
    target_size = flask.request.args.get('size', '320x240')
    tw, th = map(int, target_size.split('x'))
    frame = read_frame('s3://dexmal-sharefs-pdd/300k_official/' + path, idx)
    if frame.shape[0] * tw >= frame.shape[1] * th:
        ow = frame.shape[1] * th // frame.shape[0]
        frame = cv2.resize(frame, (ow, th))
        frame = cv2.copyMakeBorder(frame, 0, 0, (tw - ow) // 2, (tw - ow + 1) // 2, cv2.BORDER_CONSTANT, value = (255, 255, 255))
    else:
        oh = frame.shape[0] * tw // frame.shape[1]
        frame = cv2.resize(frame, (tw, oh))
        frame = cv2.copyMakeBorder(frame, (th - oh)//2, (th - oh + 1) // 2, 0, 0, cv2.BORDER_CONSTANT, value = (255, 255, 255))
    content = cv2.imencode('.jpg', frame)[1].tobytes()
    return flask.send_file(io.BytesIO(content), mimetype = 'image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5311, threaded = False, debug = True)
