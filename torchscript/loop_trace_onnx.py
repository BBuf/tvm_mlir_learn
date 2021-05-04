import torch

# Trace-based only

class LoopModel(torch.nn.Module):
    def forward(self, x, y):
        for i in range(y):
            x = x + i
        return x

model = LoopModel()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)

torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True)