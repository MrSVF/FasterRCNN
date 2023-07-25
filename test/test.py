import os
import json

meta_path = os.path.join('test/1F.json')
with open(meta_path, "r") as meta_f:
    meta_data = json.load(meta_f)

time_stamp = meta_data['data']["\u041f\u0435\u0440\u0435\u0441\u0442\u0440\u043e\u0435\u043d\u0438\u044f1F_0000000001.jpg"]
# time_stamp = float("{:.3f}".format(time_stamp))
time_stamp = float(f"{time_stamp:.3f}")
# time_stamp = round(time_stamp, 3)

print(type(time_stamp), time_stamp)