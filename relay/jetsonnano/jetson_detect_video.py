import os
import tvm
import cv2
import time
import numpy as np
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
from tvm.contrib import graph_runtime
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

show = False

set_cuda_target_arch('sm_53')
target = tvm.target.cuda(model="nano")
target_host = "llvm -target=aarch64-linux-gnu"
target_path = "/home/jetbot/zhangyiyang/data/darknet/tvm/remote"
video_path = '/home/jetbot/zhangyiyang/data/videos/Office-Parkour.mp4'

# set_cuda_target_arch('sm_70')
# target = tvm.target.cuda()
# target_host = "llvm -target=x86_64-linux-gnu"
# target_path = "/home/bbuf/tvm_tests/jetbot_tune/local"
# video_path = "/home/bbuf/data/videos/Office-Parkour.mp4"

dtype = "float32"
ctx = tvm.gpu()

names = None
if show:
    coco_path = "/home/jetbot/zhangyiyang/darknet/data/coco.names"
    # coco_path = "/home/bbuf/darknet/data/coco.names"
    with open(coco_path) as f:
        content = f.readlines()
    names = [x.strip() for x in content]


def imports(target_path, source):
    lib_path = os.path.join(target_path,
                            'yolov3-tiny-{}-lib.tar'.format(source))
    graph_path = os.path.join(target_path,
                              'yolov3-tiny-{}-graph.json'.format(source))
    params_path = os.path.join(target_path,
                               'yolov3-tiny-{}-param.params'.format(source))
    print(lib_path, graph_path, params_path)
    loaded_json = open(graph_path).read()
    loaded_lib = tvm.module.load(lib_path)
    loaded_params = bytearray(open(params_path, "rb").read())
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)

    return module


def test_tf():
    m = imports(target_path, "tf")
    img = np.random.rand(416, 416, 3)
    m.set_input('Placeholder', tvm.nd.array(img.astype("float32")))
    m.run()
    outputs = m.get_output(0).asnumpy()
    print('successfully test tf, output shape {}'.format(outputs.shape))


def test_darknet():
    m = imports(target_path, "darknet")
    img = np.random.rand(3, 416, 416)
    m.set_input('data', tvm.nd.array(img.astype("float32")))
    m.run()
    outputs = m.get_output(0).asnumpy()
    print("successfully test darknet, output shape {}".format(outputs.shape))


def darwBbox(dets, img, thresh, names):
    img2 = img * 255
    img2 = img2.astype(np.uint8)
    for det in dets:
        cat = np.argmax(det['prob'])
        if det['prob'][cat] < thresh:
            continue

        imh, imw, _ = img2.shape
        b = det['bbox']
        left = int((b.x-b.w/2.)*imw)
        right = int((b.x+b.w/2.)*imw)
        top = int((b.y-b.h/2.)*imh)
        bot = int((b.y+b.h/2.)*imh)
        pt1 = (left, top)
        pt2 = (right, bot)
        text = names[cat] + " [" + str(round(det['prob'][cat] * 100, 2)) + "]"
        cv2.rectangle(img2, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(img2,
                    text,
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img2


def evaluate():
    m = imports(target_path, "darknet")

    # run
    if show:
        cv2.namedWindow('DarkNet', cv2.WINDOW_AUTOSIZE)

    nms_thresh = 0.5
    steps = 100
    cnt = 0
    cap = cv2.VideoCapture(video_path)
    res, frame = cap.read()
    start = time.time()
    s = start
    tp_model = 0.
    tp_read = .0
    tp_preprocessing = .0

    print("start running")
    while res:
        t1 = time.time()

        # image preprocessing
        img = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
        img = np.divide(img, 255.0)
        img = img.transpose((2, 0, 1))
        img = np.flip(img, 0)
        t2 = time.time()

        # set data
        m.set_input('data', tvm.nd.array(img.astype(dtype)))
        m.run()
        tvm_out = []
        thresh = 0.560
        for i in range(2):
            layer_out = {}
            layer_out['type'] = 'Yolo'
            layer_attr = m.get_output(i*4+3).asnumpy()
            layer_out['biases'] = m.get_output(i*4+2).asnumpy()
            layer_out['mask'] = m.get_output(i*4+1).asnumpy()
            out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                         layer_attr[2], layer_attr[3])
            layer_out['output'] = m.get_output(
                i*4).asnumpy().reshape(out_shape)
            layer_out['classes'] = layer_attr[4]
            tvm_out.append(layer_out)
        t3 = time.time()

        # post ops
        dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
            (416, 416), (416, 416), thresh, 1, tvm_out)
        tvm.relay.testing.yolo_detection.do_nms_sort(
            dets, 80, nms_thresh)

        # show images
        if show:
            img = img.transpose(1, 2, 0)
            img = darwBbox(dets, img, thresh, names)
            img = np.flip(img, 2)
            cv2.imshow('DarkNet', img)
            cv2.waitKey(1)
        t4 = time.time()

        # read frame
        res, frame = cap.read()

        cnt += 1
        t5 = time.time()
        tp_model += t4-t2
        tp_read += t5-t4
        tp_preprocessing += t2-t1

        print('%05f %05f %05f %05f %05f' %
              (t2-t1, t3-t2, t4-t3, t5-t4, t5-t1))
    total_time = time.time()-s
    print(total_time/cnt, cnt/total_time, tp_read/(cnt-1),
          tp_preprocessing / cnt, tp_model / cnt)
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    evaluate()
