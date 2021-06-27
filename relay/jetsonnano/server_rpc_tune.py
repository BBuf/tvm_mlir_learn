import os

import numpy as np

import tensorflow as tf
import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.relay.testing.tf as tf_testing
import tvm.contrib.graph_runtime as runtime
from tvm.relay.testing.darknet import __darknetffi__
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
set_cuda_target_arch('sm_53')

target = tvm.target.cuda(model="nano")
target_host = "llvm -target=aarch64-linux-gnu"
network = 'jetson-yolov3-tiny-v2'
log_file = "%s.log" % network
dtype = 'float32'

source = 'darknet'


tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 200,
    'early_stopping': 100,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=1000),
        runner=autotvm.RPCRunner(
            'jetbot', '0.0.0.0', 9190,
            number=5,
            timeout=1000,
        )
    ),
}


def get_tf_yolov3_tiny(
        model_path=("/home/bbuf/Tensorflow-YOLOv3/"
                    "weights/raw-yolov3-tiny.pb"),
        outputs=['yolov3_tiny/concat_6'],):
    input_shape = (1, 416, 416, 3)

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
    return mod, params, input_shape


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

    DARKNET_LIB = __darknetffi__.dlopen(lib_path)
    net = DARKNET_LIB.load_network(cfg_path.encode(
        'utf-8'), weights_path.encode('utf-8'), 0)
    input_shape = [1, net.c, net.h, net.w]
    mod, params = relay.frontend.from_darknet(
        net, dtype='float32', shape=input_shape)
    print("successfully load darknet relay")
    return mod, params, input_shape


def get_network(source):
    if source == 'tf':
        return get_tf_yolov3_tiny()
    elif source == 'darknet':
        return get_darknet()
    raise ValueError('unknown source {}'.format(source))


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name,
                                          tasks[i].args,
                                          tasks[i].target,
                                          tasks[i].target_host,
                                          'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(
                    autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(
                               n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


use_android = False


def tune_and_evaluate(tuning_opt, source):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape = get_network(source)
    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              target_host=target_host,
                                              params=params,
                                              ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params, target_host=target_host)
        # export library
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk
            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote("jetbot",
                                                '0.0.0.0',
                                                9190,
                                                timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        ctx = remote.context(str(target), 0)
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array(
            (np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=30)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


tune_and_evaluate(tuning_option, source)
