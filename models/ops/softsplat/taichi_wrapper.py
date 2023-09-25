from .taichi_ops import softsplat_worker
import comfy.model_management as model_management
import torch
import torch.multiprocessing as mp

queue, process, recieved_event = None, None, None

device = model_management.get_torch_device()
def send_recieve_softsplat(tenIn, tenFlow):
    global queue, process, recieved_event
    if queue is None:
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()
        recieved_event = mp.Event()
        recieved_event.clear()
        process = mp.Process(target=softsplat_worker, args=(queue, recieved_event, device))
        process.start()
    
    queue.put((tenIn.cpu(), tenFlow.cpu()))
    recieved_event.wait()
    recieved_event.clear()
    return queue.get().to(model_management.get_torch_device())

def softsplat(
    tenIn: torch.Tensor, tenFlow: torch.Tensor, tenMetric: torch.Tensor, strMode: str
):
    assert strMode.split("-")[0] in ["sum", "avg", "linear", "soft"]

    if strMode == "sum":
        assert tenMetric is None
    if strMode == "avg":
        assert tenMetric is None
    if strMode.split("-")[0] == "linear":
        assert tenMetric is not None
    if strMode.split("-")[0] == "soft":
        assert tenMetric is not None

    if strMode == "avg":
        tenIn = torch.cat(
            [
                tenIn,
                tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]),
            ],
            1,
        )

    elif strMode.split("-")[0] == "linear":
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif strMode.split("-")[0] == "soft":
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    # end

    tenOut = send_recieve_softsplat(tenIn, tenFlow)

    if strMode.split("-")[0] in ["avg", "linear", "soft"]:
        tenNormalize = tenOut[:, -1:, :, :]

        if len(strMode.split("-")) == 1:
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "addeps":
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "zeroeps":
            tenNormalize[tenNormalize == 0.0] = 1.0

        elif strMode.split("-")[1] == "clipeps":
            tenNormalize = tenNormalize.clip(0.0000001, None)

        # end

        tenOut = tenOut[:, :-1, :, :] / tenNormalize
    # end

    return tenOut

def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
    assert tenMetric is None or tenMetric.shape[1] == 1
    assert strType in ["summation", "average", "linear", "softmax"]

    if strType == "average":
        tenInput = torch.cat(
            [
                tenInput,
                tenInput.new_ones(
                    tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3]
                ),
            ],
            1,
        )

    elif strType == "linear":
        tenInput = torch.cat([tenInput * tenMetric, tenMetric], 1)

    elif strType == "softmax":
        tenInput = torch.cat([tenInput * tenMetric.exp(), tenMetric.exp()], 1)

    # end

    tenOutput = send_recieve_softsplat(tenInput, tenFlow)

    if strType != "summation":
        tenNormalize = tenOutput[:, -1:, :, :]

        tenNormalize[tenNormalize == 0.0] = 1.0

        tenOutput = tenOutput[:, :-1, :, :] / tenNormalize
    # end

    return tenOutput


# end


class ModuleSoftsplat(torch.nn.Module):
    def __init__(self, strType):
        super().__init__()

        self.strType = strType

    # end

    def forward(self, tenInput, tenFlow, tenMetric):
        return FunctionSoftsplat(tenInput, tenFlow, tenMetric, self.strType)

    # end

def init():
    send_recieve_softsplat(
        torch.ones(1, 3, 256, 256, device=model_management.get_torch_device()), 
        torch.ones(1, 2, 256, 256, device=model_management.get_torch_device())
    )

__all__ = ['softsplat', 'FunctionSoftsplat', 'ModuleSoftsplat', 'init']