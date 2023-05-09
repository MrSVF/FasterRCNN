import os

# print(os.listdir('C:\\Users\\iserg\\rebels_code\\PyTorch-Object-Detection-Faster-RCNN-Tutorial\\pytorch_faster_rcnn_tutorial\\data\\speed_bump\\test2'))
str_cxcywh = ''
x1, y1, x2, y2 = 0.6142422493991063, 0.23841154393182865, 0.6311477952415451, 0.26730991289326245

cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
w = x2 - x1
h = y2 - y1

str_cxcywh+="{:.0f} ".format(0) + \
      "{:.10f} ".format(cx) + \
      "{:.10f} ".format(cy) + \
      "{:.10f} ".format(w) + \
      "{:.10f}".format(h) + '\n'

str_cxcywh+="{:.0f}".format(0) + " " +\
            "{:.10f}".format(cx).rstrip('0').rstrip('.') + " " +\
            "{:.10f}".format(cy).rstrip('0').rstrip('.') + " " +\
            "{:.10f}".format(w).rstrip('0').rstrip('.') + " " +\
            "{:.10f}".format(h).rstrip('0').rstrip('.') + '\n'

print(str_cxcywh)

str = "{:.10f}".format(0).rstrip('0').rstrip('.')
print(str)