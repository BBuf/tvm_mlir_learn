import os

import numpy as np

import tvm
import tvm.relay.testing
import tvm.relay.testing.tf as tf_testing
import tvm.relay.testing.darknet
import tvm.relay.testing.yolo_detection
from tvm import rpc
from tvm import relay
from tvm import autotvm
from tvm.contrib import graph_runtime
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
set_cuda_target_arch('sm_53')

dtype = 'float32'
input_shape = (1, 416, 416, 3)
log_file_path = "/home/bbuf/tvm_tests/jetbot_tune/jetson-yolov3-tiny-v2.log"

source = 'darknet'
use_gpu = True
local_demo = False
if local_demo:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_cuda_target_arch('sm_70')
    export_dir = "local"
    remote = rpc.LocalSession()
    target_host = "llvm"
    if use_gpu:
        target = tvm.target.cuda()
        ctx = remote.gpu()
    else:
        target = "llvm"
        ctx = remote.cpu()
else:
    host = '192.168.1.6'
    port = 9091
    export_dir = "remote"
    remote = rpc.connect(host, port)
    target_host = "llvm -target=aarch64-linux-gnu"
    if use_gpu:
        target = tvm.target.cuda()
        ctx = remote.gpu()
    else:
        target = "llvm -target=aarch64-linux-gnu"
        ctx = remote.cpu()


def get_tf_yolov3_tiny(
        model_path=("/home/bbuf/Tensorflow-YOLOv3/"
                    "weights/raw-yolov3-tiny.pb"),
        outputs=['yolov3_tiny/concat_6'],):
    import tensorflow as tf

    with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        with tf.compat.v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(
                sess, outputs[0])
        print("successfully load tf model")

    mod, params = relay.frontend.from_tensorflow(
        graph_def,
        layout="NCHW",
        shape={'Placeholder': input_shape},
        outputs=outputs,
    )
    print("successfully convert tf model to relay")
    return mod, params


def get_darknet():
    ######################################################################
    # Choose the model
    # -----------------------
    # Models are: 'yolov2', 'yolov3' or 'yolov3-tiny'

    # Model name
    MODEL_NAME = "yolov3-tiny"
    CFG_NAME = MODEL_NAME + '.cfg'
    WEIGHTS_NAME = MODEL_NAME + '.weights'
    REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
    CFG_URL = 'https://github.com/pjreddie/darknet/raw/master/cfg/' + CFG_NAME + '?raw=true'
    WEIGHTS_URL = 'https://pjreddie.com/media/files/' + WEIGHTS_NAME

    cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
    weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")

    # Download and Load darknet library
    if sys.platform in ['linux', 'linux2']:
        DARKNET_LIB = 'libdarknet2.0.so'
        DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
    elif sys.platform == 'darwin':
        DARKNET_LIB = 'libdarknet_mac2.0.so'
        DARKNET_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'
    else:
        err = "Darknet lib is not supported on {} platform".format(sys.platform)
        raise NotImplementedError(err)

    lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")
    
    from tvm.relay.testing.darknet import __darknetffi__
    DARKNET_LIB = __darknetffi__.dlopen(lib_path)
    net = DARKNET_LIB.load_network(
        cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
    mod, params = relay.frontend.from_darknet(
        net, dtype='float32', shape=[1, net.c, net.h, net.w])
    print("successfully load darknet relay")
    return mod, params


def exports(source, target_path="./exports"):
    if source == 'darknet':
        mod, params = get_darknet()
    elif source == 'tf':
        mod, params = get_tf_yolov3_tiny()
    else:
        raise ValueError('unknown source {}'.format(source))

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                         target=target,
                                         target_host=target_host,
                                         params=params)

    lib_name = 'yolov3-tiny-{}-lib.tar'.format(source)
    lib_path = os.path.join(target_path, lib_name)
    graph_path = os.path.join(target_path,
                              'yolov3-tiny-{}-graph.json'.format(source))
    params_path = os.path.join(target_path,
                               'yolov3-tiny-{}-param.params'.format(source))
    lib.export_library(lib_path)
    # loaded_lib = tvm.module.load(lib_path)
    print(lib_path)
    with open(graph_path, "w") as fo:
        fo.write(graph)
    with open(params_path, "wb") as fo:
        fo.write(relay.save_param_dict(params))

    return lib_path, graph_path, params_path, graph, params, lib_name


def detect_video(m):
    import cv2
    import time
    video_path = "/home/bbuf/data/videos/Office-Parkour.mp4"
    nms_thresh = 0.5
    thresh = 0.560
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

        if cnt % steps == 0:
            end = time.time()
            print(steps/(end-start), tp_model/cnt)
            start = end
    total_time = time.time()-s
    print(total_time/cnt, cnt/total_time, tp_read/(cnt-1),
          tp_preprocessing/cnt, tp_model/cnt)


def evaluate():
    lib_path, graph_path, params_path, loaded_json, loaded_params, lib_name\
        = exports(source, export_dir)

    remote.upload(lib_path)
    loaded_lib = remote.load_module(lib_name)
    m = graph_runtime.create(loaded_json, loaded_lib, ctx)
    m.set_input(**loaded_params)

    detect_video(m)

    # if source == 'tf':
    #     img = np.random.rand(416, 416, 3)
    #     m.set_input('Placeholder', tvm.nd.array(img.astype("float32")))
    # elif source == 'darknet':
    #     img = np.random.rand(3, 416, 416)
    #     m.set_input('data', tvm.nd.array(img.astype("float32")))

    # m.run()
    # outputs = m.get_output(0).asnumpy()
    # print("successfully test darknet, output shape {}".format(outputs.shape))


if __name__ == "__main__":
    if log_file_path is None:
        evaluate()
    else:
        with autotvm.apply_history_best(log_file_path):
            evaluate()
