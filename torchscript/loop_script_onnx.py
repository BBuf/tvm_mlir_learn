import torch
# Mixing tracing and scripting

@torch.jit.script
def loop(x, y):
    for i in range(int(y)):
        x = x + i
    return x

class LoopModel2(torch.nn.Module):
    def forward(self, x, y):
        return loop(x, y)

model = LoopModel2()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)
torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True,
                  input_names=['input_data', 'loop_range'])