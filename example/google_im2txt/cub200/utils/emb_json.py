# -*- coding:utf-8 -*-

import json


def store_json_file(filename, data):
    with open(filename, 'w') as json_file:
        json_file.write(json.dumps(data, indent=4))


def load_json_file(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        return data


if __name__ == "__main__":
    # measurements = [
    #     {'weight': 392.3, 'color': 'purple', 'temperature': 33.4},
    #     {'weight': 34.0, 'color': 'green', 'temperature': -3.1},
    # ]
    # store_json_file('data.json', measurements)

    data = load_json_file('image_caption.json')
    for item in data:
        print("COCO_val2014_%012d.jpg" % item['image_id'])

