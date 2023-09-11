import torch
from .utils import cuda_launch, cuda_kernel, cuda_int32
import cupy
import collections

kernel_Softsplat_updateOutput = """
    extern "C" __global__ void kernel_Softsplat_updateOutput(
        const int n,
        const float* input,
        const float* flow,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX = ( intIndex                                                    ) % SIZE_3(output);

        float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
        float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

        int intNorthwestX = (int) (floor(fltOutputX));
        int intNorthwestY = (int) (floor(fltOutputY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        float fltNorthwest = ((float) (intSoutheastX) - fltOutputX) * ((float) (intSoutheastY) - fltOutputY);
        float fltNortheast = (fltOutputX - (float) (intSouthwestX)) * ((float) (intSouthwestY) - fltOutputY);
        float fltSouthwest = ((float) (intNortheastX) - fltOutputX) * (fltOutputY - (float) (intNortheastY));
        float fltSoutheast = (fltOutputX - (float) (intNorthwestX)) * (fltOutputY - (float) (intNorthwestY));

        if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intNorthwestY, intNorthwestX)], VALUE_4(input, intN, intC, intY, intX) * fltNorthwest);
        }

        if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intNortheastY, intNortheastX)], VALUE_4(input, intN, intC, intY, intX) * fltNortheast);
        }

        if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intSouthwestY, intSouthwestX)], VALUE_4(input, intN, intC, intY, intX) * fltSouthwest);
        }

        if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intSoutheastY, intSoutheastX)], VALUE_4(input, intN, intC, intY, intX) * fltSoutheast);
        }
    } }
"""

kernel_Softsplat_updateGradInput = """
    extern "C" __global__ void kernel_Softsplat_updateGradInput(
        const int n,
        const float* input,
        const float* flow,
        const float* gradOutput,
        float* gradInput,
        float* gradFlow
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput) / SIZE_1(gradInput) ) % SIZE_0(gradInput);
        const int intC = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput)                     ) % SIZE_1(gradInput);
        const int intY = ( intIndex / SIZE_3(gradInput)                                         ) % SIZE_2(gradInput);
        const int intX = ( intIndex                                                             ) % SIZE_3(gradInput);

        float fltGradInput = 0.0;

        float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
        float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

        int intNorthwestX = (int) (floor(fltOutputX));
        int intNorthwestY = (int) (floor(fltOutputY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        float fltNorthwest = ((float) (intSoutheastX) - fltOutputX) * ((float) (intSoutheastY) - fltOutputY);
        float fltNortheast = (fltOutputX - (float) (intSouthwestX)) * ((float) (intSouthwestY) - fltOutputY);
        float fltSouthwest = ((float) (intNortheastX) - fltOutputX) * (fltOutputY - (float) (intNortheastY));
        float fltSoutheast = (fltOutputX - (float) (intNorthwestX)) * (fltOutputY - (float) (intNorthwestY));

        if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(gradOutput)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(gradOutput))) {
            fltGradInput += VALUE_4(gradOutput, intN, intC, intNorthwestY, intNorthwestX) * fltNorthwest;
        }

        if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(gradOutput)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(gradOutput))) {
            fltGradInput += VALUE_4(gradOutput, intN, intC, intNortheastY, intNortheastX) * fltNortheast;
        }

        if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(gradOutput)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(gradOutput))) {
            fltGradInput += VALUE_4(gradOutput, intN, intC, intSouthwestY, intSouthwestX) * fltSouthwest;
        }

        if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(gradOutput)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(gradOutput))) {
            fltGradInput += VALUE_4(gradOutput, intN, intC, intSoutheastY, intSoutheastX) * fltSoutheast;
        }

        gradInput[intIndex] = fltGradInput;
    } }
"""

kernel_Softsplat_updateGradFlow = """
    extern "C" __global__ void kernel_Softsplat_updateGradFlow(
        const int n,
        const float* input,
        const float* flow,
        const float* gradOutput,
        float* gradInput,
        float* gradFlow
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float fltGradFlow = 0.0;

        const int intN = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow) / SIZE_1(gradFlow) ) % SIZE_0(gradFlow);
        const int intC = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow)                    ) % SIZE_1(gradFlow);
        const int intY = ( intIndex / SIZE_3(gradFlow)                                       ) % SIZE_2(gradFlow);
        const int intX = ( intIndex                                                          ) % SIZE_3(gradFlow);

        float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
        float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

        int intNorthwestX = (int) (floor(fltOutputX));
        int intNorthwestY = (int) (floor(fltOutputY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        float fltNorthwest = 0.0;
        float fltNortheast = 0.0;
        float fltSouthwest = 0.0;
        float fltSoutheast = 0.0;

        if (intC == 0) {
            fltNorthwest = ((float) (-1.0)) * ((float) (intSoutheastY) - fltOutputY);
            fltNortheast = ((float) (+1.0)) * ((float) (intSouthwestY) - fltOutputY);
            fltSouthwest = ((float) (-1.0)) * (fltOutputY - (float) (intNortheastY));
            fltSoutheast = ((float) (+1.0)) * (fltOutputY - (float) (intNorthwestY));

        } else if (intC == 1) {
            fltNorthwest = ((float) (intSoutheastX) - fltOutputX) * ((float) (-1.0));
            fltNortheast = (fltOutputX - (float) (intSouthwestX)) * ((float) (-1.0));
            fltSouthwest = ((float) (intNortheastX) - fltOutputX) * ((float) (+1.0));
            fltSoutheast = (fltOutputX - (float) (intNorthwestX)) * ((float) (+1.0));

        }

        for (int intChannel = 0; intChannel < SIZE_1(gradOutput); intChannel += 1) {
            float fltInput = VALUE_4(input, intN, intChannel, intY, intX);

            if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(gradOutput)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(gradOutput))) {
                fltGradFlow += fltInput * VALUE_4(gradOutput, intN, intChannel, intNorthwestY, intNorthwestX) * fltNorthwest;
            }

            if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(gradOutput)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(gradOutput))) {
                fltGradFlow += fltInput * VALUE_4(gradOutput, intN, intChannel, intNortheastY, intNortheastX) * fltNortheast;
            }

            if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(gradOutput)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(gradOutput))) {
                fltGradFlow += fltInput * VALUE_4(gradOutput, intN, intChannel, intSouthwestY, intSouthwestX) * fltSouthwest;
            }

            if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(gradOutput)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(gradOutput))) {
                fltGradFlow += fltInput * VALUE_4(gradOutput, intN, intChannel, intSoutheastY, intSoutheastX) * fltSoutheast;
            }
        }

        gradFlow[intIndex] = fltGradFlow;
    } }
"""

softsplat_flowgrad = """
    extern "C" __global__ void __launch_bounds__(512) softsplat_flowgrad(
        const int n,
        const {{type}}* __restrict__ tenIn,
        const {{type}}* __restrict__ tenFlow,
        const {{type}}* __restrict__ tenOutgrad,
        {{type}}* __restrict__ tenIngrad,
        {{type}}* __restrict__ tenFlowgrad
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenFlowgrad) / SIZE_2(tenFlowgrad) / SIZE_1(tenFlowgrad) ) % SIZE_0(tenFlowgrad);
        const int intC = ( intIndex / SIZE_3(tenFlowgrad) / SIZE_2(tenFlowgrad)                       ) % SIZE_1(tenFlowgrad);
        const int intY = ( intIndex / SIZE_3(tenFlowgrad)                                             ) % SIZE_2(tenFlowgrad);
        const int intX = ( intIndex                                                                   ) % SIZE_3(tenFlowgrad);

        assert(SIZE_1(tenFlow) == 2);

        {{type}} fltFlowgrad = 0.0f;

        {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
        {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

        if (isfinite(fltX) == false) { return; }
        if (isfinite(fltY) == false) { return; }

        int intNorthwestX = (int) (floor(fltX));
        int intNorthwestY = (int) (floor(fltY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        {{type}} fltNorthwest = 0.0f;
        {{type}} fltNortheast = 0.0f;
        {{type}} fltSouthwest = 0.0f;
        {{type}} fltSoutheast = 0.0f;

        if (intC == 0) {
            fltNorthwest = (({{type}}) (-1.0f)) * (({{type}}) (intSoutheastY) - fltY);
            fltNortheast = (({{type}}) (+1.0f)) * (({{type}}) (intSouthwestY) - fltY);
            fltSouthwest = (({{type}}) (-1.0f)) * (fltY - ({{type}}) (intNortheastY));
            fltSoutheast = (({{type}}) (+1.0f)) * (fltY - ({{type}}) (intNorthwestY));

        } else if (intC == 1) {
            fltNorthwest = (({{type}}) (intSoutheastX) - fltX) * (({{type}}) (-1.0f));
            fltNortheast = (fltX - ({{type}}) (intSouthwestX)) * (({{type}}) (-1.0f));
            fltSouthwest = (({{type}}) (intNortheastX) - fltX) * (({{type}}) (+1.0f));
            fltSoutheast = (fltX - ({{type}}) (intNorthwestX)) * (({{type}}) (+1.0f));

        }

        for (int intChannel = 0; intChannel < SIZE_1(tenOutgrad); intChannel += 1) {
            {{type}} fltIn = VALUE_4(tenIn, intN, intChannel, intY, intX);

            if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(tenOutgrad)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(tenOutgrad))) {
                fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intNorthwestY, intNorthwestX) * fltIn * fltNorthwest;
            }

            if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(tenOutgrad)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(tenOutgrad))) {
                fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intNortheastY, intNortheastX) * fltIn * fltNortheast;
            }

            if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(tenOutgrad)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(tenOutgrad))) {
                fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intSouthwestY, intSouthwestX) * fltIn * fltSouthwest;
            }

            if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(tenOutgrad)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(tenOutgrad))) {
                fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intSoutheastY, intSoutheastX) * fltIn * fltSoutheast;
            }
        }

        tenFlowgrad[intIndex] = fltFlowgrad;
    } }
"""

softsplat_ingrad = """
    extern "C" __global__ void __launch_bounds__(512) softsplat_ingrad(
        const int n,
        const {{type}}* __restrict__ tenIn,
        const {{type}}* __restrict__ tenFlow,
        const {{type}}* __restrict__ tenOutgrad,
        {{type}}* __restrict__ tenIngrad,
        {{type}}* __restrict__ tenFlowgrad
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad) / SIZE_1(tenIngrad) ) % SIZE_0(tenIngrad);
        const int intC = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad)                     ) % SIZE_1(tenIngrad);
        const int intY = ( intIndex / SIZE_3(tenIngrad)                                         ) % SIZE_2(tenIngrad);
        const int intX = ( intIndex                                                             ) % SIZE_3(tenIngrad);

        assert(SIZE_1(tenFlow) == 2);

        {{type}} fltIngrad = 0.0f;

        {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
        {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

        if (isfinite(fltX) == false) { return; }
        if (isfinite(fltY) == false) { return; }

        int intNorthwestX = (int) (floor(fltX));
        int intNorthwestY = (int) (floor(fltY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        {{type}} fltNorthwest = (({{type}}) (intSoutheastX) - fltX) * (({{type}}) (intSoutheastY) - fltY);
        {{type}} fltNortheast = (fltX - ({{type}}) (intSouthwestX)) * (({{type}}) (intSouthwestY) - fltY);
        {{type}} fltSouthwest = (({{type}}) (intNortheastX) - fltX) * (fltY - ({{type}}) (intNortheastY));
        {{type}} fltSoutheast = (fltX - ({{type}}) (intNorthwestX)) * (fltY - ({{type}}) (intNorthwestY));

        if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(tenOutgrad)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(tenOutgrad))) {
            fltIngrad += VALUE_4(tenOutgrad, intN, intC, intNorthwestY, intNorthwestX) * fltNorthwest;
        }

        if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(tenOutgrad)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(tenOutgrad))) {
            fltIngrad += VALUE_4(tenOutgrad, intN, intC, intNortheastY, intNortheastX) * fltNortheast;
        }

        if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(tenOutgrad)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(tenOutgrad))) {
            fltIngrad += VALUE_4(tenOutgrad, intN, intC, intSouthwestY, intSouthwestX) * fltSouthwest;
        }

        if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(tenOutgrad)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(tenOutgrad))) {
            fltIngrad += VALUE_4(tenOutgrad, intN, intC, intSoutheastY, intSoutheastX) * fltSoutheast;
        }

        tenIngrad[intIndex] = fltIngrad;
    } }
"""

softsplat_out = """
    extern "C" __global__ void __launch_bounds__(512) softsplat_out(
        const int n,
        const {{type}}* __restrict__ tenIn,
        const {{type}}* __restrict__ tenFlow,
        {{type}}* __restrict__ tenOut
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut) / SIZE_1(tenOut) ) % SIZE_0(tenOut);
        const int intC = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut)                  ) % SIZE_1(tenOut);
        const int intY = ( intIndex / SIZE_3(tenOut)                                   ) % SIZE_2(tenOut);
        const int intX = ( intIndex                                                    ) % SIZE_3(tenOut);

        assert(SIZE_1(tenFlow) == 2);

        {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
        {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

        if (isfinite(fltX) == false) { return; }
        if (isfinite(fltY) == false) { return; }

        {{type}} fltIn = VALUE_4(tenIn, intN, intC, intY, intX);

        int intNorthwestX = (int) (floor(fltX));
        int intNorthwestY = (int) (floor(fltY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        {{type}} fltNorthwest = (({{type}}) (intSoutheastX) - fltX) * (({{type}}) (intSoutheastY) - fltY);
        {{type}} fltNortheast = (fltX - ({{type}}) (intSouthwestX)) * (({{type}}) (intSouthwestY) - fltY);
        {{type}} fltSouthwest = (({{type}}) (intNortheastX) - fltX) * (fltY - ({{type}}) (intNortheastY));
        {{type}} fltSoutheast = (fltX - ({{type}}) (intNorthwestX)) * (fltY - ({{type}}) (intNorthwestY));

        if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(tenOut)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(tenOut))) {
            atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intNorthwestY, intNorthwestX)], fltIn * fltNorthwest);
        }

        if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(tenOut)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(tenOut))) {
            atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intNortheastY, intNortheastX)], fltIn * fltNortheast);
        }

        if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(tenOut)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(tenOut))) {
            atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intSouthwestY, intSouthwestX)], fltIn * fltSouthwest);
        }

        if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(tenOut)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(tenOut))) {
            atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intSoutheastY, intSoutheastX)], fltIn * fltSoutheast);
        }
    } }
"""

class _FunctionSoftsplat(torch.autograd.Function):
    @staticmethod
    def forward(self, input, flow):
        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = (
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )
        intFlowDepth, intFlowHeight, intFlowWidth = (
            flow.shape[1],
            flow.shape[2],
            flow.shape[3],
        )

        assert intFlowDepth == 2
        assert intInputHeight == intFlowHeight
        assert intInputWidth == intFlowWidth

        input = input.contiguous()
        assert input.is_cuda == True
        flow = flow.contiguous()
        assert flow.is_cuda == True

        output = input.new_zeros(
            [intSamples, intInputDepth, intInputHeight, intInputWidth]
        )

        if input.is_cuda == True:
            n = output.nelement()
            cuda_launch(
                cuda_kernel(
                    "kernel_Softsplat_updateOutput",
                    kernel_Softsplat_updateOutput,
                    {"input": input, "flow": flow, "output": output},
                ),
            )(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cupy.int32(n),
                    input.data_ptr(),
                    flow.data_ptr(),
                    output.data_ptr(),
                ],
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        self.save_for_backward(input, flow)

        return output

    # end

    @staticmethod
    def backward(self, gradOutput):
        input, flow = self.saved_tensors

        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = (
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )
        intFlowDepth, intFlowHeight, intFlowWidth = (
            flow.shape[1],
            flow.shape[2],
            flow.shape[3],
        )

        assert intFlowDepth == 2
        assert intInputHeight == intFlowHeight
        assert intInputWidth == intFlowWidth

        gradOutput = gradOutput.contiguous()
        assert gradOutput.is_cuda == True

        gradInput = (
            input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth])
            if self.needs_input_grad[0] == True
            else None
        )
        gradFlow = (
            input.new_zeros([intSamples, intFlowDepth, intFlowHeight, intFlowWidth])
            if self.needs_input_grad[1] == True
            else None
        )

        if input.is_cuda == True:
            if gradInput is not None:
                n = gradInput.nelement()
                cuda_launch(
                    "kernel_Softsplat_updateGradInput",
                    cuda_kernel(
                        "kernel_Softsplat_updateGradInput",
                        kernel_Softsplat_updateGradInput,
                        {
                            "input": input,
                            "flow": flow,
                            "gradOutput": gradOutput,
                            "gradInput": gradInput,
                            "gradFlow": gradFlow,
                        },
                    ),
                )(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[
                        cupy.int32(n),
                        input.data_ptr(),
                        flow.data_ptr(),
                        gradOutput.data_ptr(),
                        gradInput.data_ptr(),
                        None,
                    ],
                )
            # end

            if gradFlow is not None:
                n = gradFlow.nelement()
                cuda_launch(
                    cuda_kernel(
                        "kernel_Softsplat_updateGradFlow",
                        kernel_Softsplat_updateGradFlow,
                        {
                            "input": input,
                            "flow": flow,
                            "gradOutput": gradOutput,
                            "gradInput": gradInput,
                            "gradFlow": gradFlow,
                        }
                    )
                )(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[
                        cupy.int32(n),
                        input.data_ptr(),
                        flow.data_ptr(),
                        gradOutput.data_ptr(),
                        None,
                        gradFlow.data_ptr(),
                    ],
                )
            # end

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return gradInput, gradFlow

    # end


# end

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

    tenOutput = _FunctionSoftsplat.apply(tenInput, tenFlow)

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


# end



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

    tenOut = softsplat_func.apply(tenIn, tenFlow)

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


# end


class softsplat_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tenIn, tenFlow):
        tenOut = tenIn.new_zeros(
            [tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]]
        )

        if tenIn.is_cuda == True:
            cuda_launch(
                cuda_kernel(
                    "softsplat_out",
                    softsplat_out,
                    {"tenIn": tenIn, "tenFlow": tenFlow, "tenOut": tenOut},
                )
            )(
                grid=tuple([int((tenOut.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenOut.nelement()),
                    tenIn.data_ptr(),
                    tenFlow.data_ptr(),
                    tenOut.data_ptr(),
                ],
                stream=collections.namedtuple("Stream", "ptr")(
                    torch.cuda.current_stream().cuda_stream
                ),
            )

        elif tenIn.is_cuda != True:
            assert False

        # end

        self.save_for_backward(tenIn, tenFlow)

        return tenOut

    # end

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, tenOutgrad):
        tenIn, tenFlow = self.saved_tensors

        tenOutgrad = tenOutgrad.contiguous()
        assert tenOutgrad.is_cuda == True

        tenIngrad = (
            tenIn.new_zeros(
                [tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]]
            )
            if self.needs_input_grad[0] == True
            else None
        )
        tenFlowgrad = (
            tenFlow.new_zeros(
                [tenFlow.shape[0], tenFlow.shape[1], tenFlow.shape[2], tenFlow.shape[3]]
            )
            if self.needs_input_grad[1] == True
            else None
        )

        if tenIngrad is not None:
            cuda_launch(
                cuda_kernel(
                    "softsplat_ingrad",
                    softsplat_ingrad,
                    {
                        "tenIn": tenIn,
                        "tenFlow": tenFlow,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenFlowgrad": tenFlowgrad,
                    },
                )
            )(
                grid=tuple([int((tenIngrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenIngrad.nelement()),
                    tenIn.data_ptr(),
                    tenFlow.data_ptr(),
                    tenOutgrad.data_ptr(),
                    tenIngrad.data_ptr(),
                    None,
                ],
                stream=collections.namedtuple("Stream", "ptr")(
                    torch.cuda.current_stream().cuda_stream
                ),
            )
        # end

        if tenFlowgrad is not None:
            cuda_launch(
                cuda_kernel(
                    "softsplat_flowgrad",
                    softsplat_flowgrad,
                    {
                        "tenIn": tenIn,
                        "tenFlow": tenFlow,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenFlowgrad": tenFlowgrad,
                    },
                )
            )(
                grid=tuple([int((tenFlowgrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenFlowgrad.nelement()),
                    tenIn.data_ptr(),
                    tenFlow.data_ptr(),
                    tenOutgrad.data_ptr(),
                    None,
                    tenFlowgrad.data_ptr(),
                ],
                stream=collections.namedtuple("Stream", "ptr")(
                    torch.cuda.current_stream().cuda_stream
                ),
            )
        # end

        return tenIngrad, tenFlowgrad

    # end


# end