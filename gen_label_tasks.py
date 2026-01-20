import redis
import megfile
import random
import json
import tqdm

rd = redis.StrictRedis('localhost')

robot_name = 'aloha_5'
data_dir_base = 's3://dexmal-sharefs-pdd/300k_official'

if __name__ == '__main__':
    all_paths = []
    print('scanning all files')
    for path in megfile.smart_scan(data_dir_base + '/' + robot_name):
        if path.endswith('high_rgb.mp4'):
            all_paths.append(path[len(data_dir_base + '/'):][:-len('/videos/cam_high_rgb.mp4')])
    rnd = random.Random(100)
    selected = []
    for i in tqdm.tqdm(range(1000)):
        path = rnd.choice(all_paths)
        meta = json.loads(megfile.smart_open(data_dir_base + '/' + path + '/meta.json', 'rb').read())
        if meta['task_meta']['episode_status'] != 'LABELLED':
            continue
        total_frames = meta['task_meta']['frames']
        selected.append({'path':path, 'frameid' : rnd.randint(0, total_frames - 1)})
    print(selected[:10])
    rd.set('label_task_list_' + robot_name, json.dumps(selected))
