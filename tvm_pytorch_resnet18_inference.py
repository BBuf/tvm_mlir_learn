import time
import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata
import torch
import torchvision
# device = torch.device("cpu")
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
print(img_path)
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms


my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
target = "llvm"
target_host = "llvm"
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=7):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.
from tvm.contrib import graph_executor

m = graph_executor.GraphModule(lib["default"](dev))

tvm_time_spent=[]
torch_time_spent=[]
n_warmup=5
n_time=10
# tvm_t0 = time.process_time()
for i in range(n_warmup+n_time):
    dtype = "float32"
    # Set inputs
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    tvm_t0 = time.time()
    # Execute
    m.run()
    # Get outputs
    tvm_output = m.get_output(0)
    tvm_time_spent.append(time.time() - tvm_t0)
# tvm_t1 = time.process_time()

#####################################################################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
synset_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt",
    ]
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

class_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_classes.txt",
    ]
)
class_name = "imagenet_classes.txt"
class_path = download_testdata(class_url, class_name, module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.asnumpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# Convert input to PyTorch variable and get PyTorch result for comparison
# torch_t0 = time.process_time()
for i in range(n_warmup+n_time):
    with torch.no_grad():
        torch_img = torch.from_numpy(img)
        torch_t0 = time.time()
        output = model(torch_img)
        torch_time_spent.append(time.time() - torch_t0)
        # Get top-1 result for PyTorch
        top1_torch = np.argmax(output.numpy())
        torch_class_key = class_id_to_key[top1_torch]
# torch_t1 = time.process_time()

# tvm_time = tvm_t1 - tvm_t0
# torch_time = torch_t1 - torch_t0
tvm_time = np.mean(tvm_time_spent[n_warmup:]) * 1000
torch_time = np.mean(torch_time_spent[n_warmup:]) * 1000

print("Relay top-1 id: {}, class name: {}, class logit: {}".format(top1_tvm, key_to_classname[tvm_class_key], tvm_output.asnumpy()[0][top1_tvm]))
print("Torch top-1 id: {}, class name: {}, class logit: {}".format(top1_torch, key_to_classname[torch_class_key], output.numpy()[0][top1_torch]))
print('Relay time(ms): {:.3f}'.format(tvm_time))
print('Torch time(ms): {:.3f}'.format(torch_time))