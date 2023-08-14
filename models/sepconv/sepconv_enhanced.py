"""
23-nov-21
https://github.com/sniklaus/revisiting-sepconv/blob/fea509d98157170df1fb35bf615bd41d98858e1a/run.py
https://github.com/sniklaus/revisiting-sepconv/blob/fea509d98157170df1fb35bf615bd41d98858e1a/sepconv/sepconv.py
Deleted stuffs about arguments_strModel and getopt
"""
#!/usr/bin/env python

import cupy
import os
import re
import torch
import typing

##########################################################


objCudacache = {}


def cuda_int32(intIn: int):
    return cupy.int32(intIn)


# end


def cuda_float32(fltIn: float):
    return cupy.float32(fltIn)


# end


def cuda_kernel(strFunction: str, strKernel: str, objVariables: typing.Dict):
    if "device" not in objCudacache:
        objCudacache["device"] = torch.cuda.get_device_name()
    # end

    strKey = strFunction

    for strVariable in objVariables:
        objValue = objVariables[strVariable]

        strKey += strVariable

        if objValue is None:
            continue

        elif type(objValue) == int:
            strKey += str(objValue)

        elif type(objValue) == float:
            strKey += str(objValue)

        elif type(objValue) == bool:
            strKey += str(objValue)

        elif type(objValue) == str:
            strKey += objValue

        elif type(objValue) == torch.Tensor:
            strKey += str(objValue.dtype)
            strKey += str(objValue.shape)
            strKey += str(objValue.stride())

        elif True:
            print(strVariable, type(objValue))
            assert False

        # end
    # end

    strKey += objCudacache["device"]

    if strKey not in objCudacache:
        for strVariable in objVariables:
            objValue = objVariables[strVariable]

            if objValue is None:
                continue

            elif type(objValue) == int:
                strKernel = strKernel.replace("{{" + strVariable + "}}", str(objValue))

            elif type(objValue) == float:
                strKernel = strKernel.replace("{{" + strVariable + "}}", str(objValue))

            elif type(objValue) == bool:
                strKernel = strKernel.replace("{{" + strVariable + "}}", str(objValue))

            elif type(objValue) == str:
                strKernel = strKernel.replace("{{" + strVariable + "}}", objValue)

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.uint8:
                strKernel = strKernel.replace("{{type}}", "unsigned char")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float16:
                strKernel = strKernel.replace("{{type}}", "half")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float32:
                strKernel = strKernel.replace("{{type}}", "float")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float64:
                strKernel = strKernel.replace("{{type}}", "double")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int32:
                strKernel = strKernel.replace("{{type}}", "int")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int64:
                strKernel = strKernel.replace("{{type}}", "long")

            elif type(objValue) == torch.Tensor:
                print(strVariable, objValue.dtype)
                assert False

            elif True:
                print(strVariable, type(objValue))
                assert False

            # end
        # end

        while True:
            objMatch = re.search("(SIZE_)([0-4])(\()([^\)]*)(\))", strKernel)

            if objMatch is None:
                break
            # end

            intArg = int(objMatch.group(2))

            strTensor = objMatch.group(4)
            intSizes = objVariables[strTensor].size()

            strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
        # end

        while True:
            objMatch = re.search("(VALUE_)([0-4])(\()", strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == "(" else 0
                intParentheses -= 1 if strKernel[intStop] == ")" else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(",")

            assert intArgs == len(strArgs) - 1

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append(
                    "(("
                    + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip()
                    + ")*"
                    + str(intStrides[intArg])
                    + ")"
                )
            # end

            strKernel = strKernel.replace(
                "VALUE_" + str(intArgs) + "(" + strKernel[intStart:intStop] + ")",
                strTensor + "[" + str.join("+", strIndex) + "]",
            )
        # end

        objCudacache[strKey] = {"strFunction": strFunction, "strKernel": strKernel}
    # end

    return strKey


# end


@cupy.memoize(for_each_device=True)
def cuda_launch(strKey: str):
    if "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = "/usr/local/cuda/"
    # end

    # print(objCudacache[strKey]['strKernel'])
    # return cupy.cuda.compile_with_cache(objCudacache[strKey]['strKernel'], tuple(['-I ' + os.environ['CUDA_HOME'], '-I ' + os.environ['CUDA_HOME'] + '/include'])).get_function(objCudacache[strKey]['strFunction'])
    return cupy.RawModule(code=objCudacache[strKey]["strKernel"]).get_function(
        objCudacache[strKey]["strFunction"]
    )


# end


##########################################################


class sepconv_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tenIn, tenVer, tenHor):
        tenOut = tenIn.new_empty(
            [
                tenIn.shape[0],
                tenIn.shape[1],
                tenVer.shape[2] and tenHor.shape[2],
                tenVer.shape[3] and tenHor.shape[3],
            ]
        )

        if tenIn.is_cuda == True:
            cuda_launch(
                cuda_kernel(
                    "sepconv_out",
                    """
                extern "C" __global__ void __launch_bounds__(512) sepconv_out(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenVer,
                    const {{type}}* __restrict__ tenHor,
                    {{type}}* __restrict__ tenOut
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut) / SIZE_1(tenOut) ) % SIZE_0(tenOut);
                    const int intC = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut)                  ) % SIZE_1(tenOut);
                    const int intY = ( intIndex / SIZE_3(tenOut)                                   ) % SIZE_2(tenOut);
                    const int intX = ( intIndex                                                    ) % SIZE_3(tenOut);

                    {{type}} fltOut = 0.0;

                    {{type}} fltKahanc = 0.0;
                    {{type}} fltKahany = 0.0;
                    {{type}} fltKahant = 0.0;

                    for (int intFy = 0; intFy < SIZE_1(tenVer); intFy += 1) {
                        for (int intFx = 0; intFx < SIZE_1(tenHor); intFx += 1) {
                            fltKahany = VALUE_4(tenIn, intN, intC, intY + intFy, intX + intFx) * VALUE_4(tenVer, intN, intFy, intY, intX) * VALUE_4(tenHor, intN, intFx, intY, intX);
                            fltKahany = fltKahany - fltKahanc;
                            fltKahant = fltOut + fltKahany;
                            fltKahanc = (fltKahant - fltOut) - fltKahany;
                            fltOut = fltKahant;
                        }
                    }

                    tenOut[intIndex] = fltOut;
                } }
            """,
                    {
                        "tenIn": tenIn,
                        "tenVer": tenVer,
                        "tenHor": tenHor,
                        "tenOut": tenOut,
                    },
                )
            )(
                grid=tuple([int((tenOut.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenOut.nelement()),
                    tenIn.data_ptr(),
                    tenVer.data_ptr(),
                    tenHor.data_ptr(),
                    tenOut.data_ptr(),
                ],
            )

        elif tenIn.is_cuda != True:
            assert False

        # end

        self.save_for_backward(tenIn, tenVer, tenHor)

        return tenOut

    # end

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, tenOutgrad):
        tenIn, tenVer, tenHor = self.saved_tensors

        tenOutgrad = tenOutgrad.contiguous()
        assert tenOutgrad.is_cuda == True

        tenIngrad = (
            tenIn.new_empty(
                [tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]]
            )
            if self.needs_input_grad[0] == True
            else None
        )
        tenVergrad = (
            tenVer.new_empty(
                [tenVer.shape[0], tenVer.shape[1], tenVer.shape[2], tenVer.shape[3]]
            )
            if self.needs_input_grad[1] == True
            else None
        )
        tenHorgrad = (
            tenHor.new_empty(
                [tenHor.shape[0], tenHor.shape[1], tenHor.shape[2], tenHor.shape[3]]
            )
            if self.needs_input_grad[2] == True
            else None
        )

        if tenIngrad is not None:
            cuda_launch(
                cuda_kernel(
                    "sepconv_ingrad",
                    """
                extern "C" __global__ void __launch_bounds__(512) sepconv_ingrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenVer,
                    const {{type}}* __restrict__ tenHor,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenVergrad,
                    {{type}}* __restrict__ tenHorgrad
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad) / SIZE_1(tenIngrad) ) % SIZE_0(tenIngrad);
                    const int intC = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad)                     ) % SIZE_1(tenIngrad);
                    const int intY = ( intIndex / SIZE_3(tenIngrad)                                         ) % SIZE_2(tenIngrad);
                    const int intX = ( intIndex                                                             ) % SIZE_3(tenIngrad);

                    {{type}} fltIngrad = 0.0;

                    {{type}} fltKahanc = 0.0;
                    {{type}} fltKahany = 0.0;
                    {{type}} fltKahant = 0.0;

                    for (int intFy = 0; intFy < SIZE_1(tenVer); intFy += 1) {
                        int intKy = intY + intFy - (SIZE_1(tenVer) - 1);

                        if (intKy < 0) { continue; }
                        if (intKy >= SIZE_2(tenVer)) { continue; }

                        for (int intFx = 0; intFx < SIZE_1(tenHor); intFx += 1) {
                            int intKx = intX + intFx - (SIZE_1(tenHor) - 1);

                            if (intKx < 0) { continue; }
                            if (intKx >= SIZE_3(tenHor)) { continue; }

                            fltKahany = VALUE_4(tenVer, intN, (SIZE_1(tenVer) - 1) - intFy, intKy, intKx) * VALUE_4(tenHor, intN, (SIZE_1(tenHor) - 1) - intFx, intKy, intKx) * VALUE_4(tenOutgrad, intN, intC, intKy, intKx);
                            fltKahany = fltKahany - fltKahanc;
                            fltKahant = fltIngrad + fltKahany;
                            fltKahanc = (fltKahant - fltIngrad) - fltKahany;
                            fltIngrad = fltKahant;
                        }
                    }

                    tenIngrad[intIndex] = fltIngrad;
                } }
            """,
                    {
                        "tenIn": tenIn,
                        "tenVer": tenVer,
                        "tenHor": tenHor,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenVergrad": tenVergrad,
                        "tenHorgrad": tenHorgrad,
                    },
                )
            )(
                grid=tuple([int((tenIngrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenIngrad.nelement()),
                    tenIn.data_ptr(),
                    tenVer.data_ptr(),
                    tenHor.data_ptr(),
                    tenOutgrad.data_ptr(),
                    tenIngrad.data_ptr(),
                    None,
                    None,
                ],
            )
        # end

        if tenVergrad is not None:
            cuda_launch(
                cuda_kernel(
                    "sepconv_vergrad",
                    """
                extern "C" __global__ void __launch_bounds__(512) sepconv_vergrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenVer,
                    const {{type}}* __restrict__ tenHor,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenVergrad,
                    {{type}}* __restrict__ tenHorgrad
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenVergrad) / SIZE_2(tenVergrad) / SIZE_1(tenVergrad) ) % SIZE_0(tenVergrad);
                    const int intC = ( intIndex / SIZE_3(tenVergrad) / SIZE_2(tenVergrad)                      ) % SIZE_1(tenVergrad);
                    const int intY = ( intIndex / SIZE_3(tenVergrad)                                           ) % SIZE_2(tenVergrad);
                    const int intX = ( intIndex                                                                ) % SIZE_3(tenVergrad);

                    {{type}} fltVergrad = 0.0;

                    {{type}} fltKahanc = 0.0;
                    {{type}} fltKahany = 0.0;
                    {{type}} fltKahant = 0.0;

                    for (int intI = 0; intI < SIZE_1(tenIn); intI += 1) {
                        for (int intFx = 0; intFx < SIZE_1(tenHor); intFx += 1) {
                            fltKahany = VALUE_4(tenHor, intN, intFx, intY, intX) * VALUE_4(tenIn, intN, intI, intY + intC, intX + intFx) * VALUE_4(tenOutgrad, intN, intI, intY, intX);
                            fltKahany = fltKahany - fltKahanc;
                            fltKahant = fltVergrad + fltKahany;
                            fltKahanc = (fltKahant - fltVergrad) - fltKahany;
                            fltVergrad = fltKahant;
                        }
                    }

                    tenVergrad[intIndex] = fltVergrad;
                } }
            """,
                    {
                        "tenIn": tenIn,
                        "tenVer": tenVer,
                        "tenHor": tenHor,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenVergrad": tenVergrad,
                        "tenHorgrad": tenHorgrad,
                    },
                )
            )(
                grid=tuple([int((tenVergrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenVergrad.nelement()),
                    tenIn.data_ptr(),
                    tenVer.data_ptr(),
                    tenHor.data_ptr(),
                    tenOutgrad.data_ptr(),
                    None,
                    tenVergrad.data_ptr(),
                    None,
                ],
            )
        # end

        if tenHorgrad is not None:
            cuda_launch(
                cuda_kernel(
                    "sepconv_horgrad",
                    """
                extern "C" __global__ void __launch_bounds__(512) sepconv_horgrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenVer,
                    const {{type}}* __restrict__ tenHor,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenVergrad,
                    {{type}}* __restrict__ tenHorgrad
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenHorgrad) / SIZE_2(tenHorgrad) / SIZE_1(tenHorgrad) ) % SIZE_0(tenHorgrad);
                    const int intC = ( intIndex / SIZE_3(tenHorgrad) / SIZE_2(tenHorgrad)                      ) % SIZE_1(tenHorgrad);
                    const int intY = ( intIndex / SIZE_3(tenHorgrad)                                           ) % SIZE_2(tenHorgrad);
                    const int intX = ( intIndex                                                                ) % SIZE_3(tenHorgrad);

                    {{type}} fltHorgrad = 0.0;

                    {{type}} fltKahanc = 0.0;
                    {{type}} fltKahany = 0.0;
                    {{type}} fltKahant = 0.0;

                    for (int intI = 0; intI < SIZE_1(tenIn); intI += 1) {
                        for (int intFy = 0; intFy < SIZE_1(tenVer); intFy += 1) {
                            fltKahany = VALUE_4(tenVer, intN, intFy, intY, intX) * VALUE_4(tenIn, intN, intI, intY + intFy, intX + intC) * VALUE_4(tenOutgrad, intN, intI, intY, intX);
                            fltKahany = fltKahany - fltKahanc;
                            fltKahant = fltHorgrad + fltKahany;
                            fltKahanc = (fltKahant - fltHorgrad) - fltKahany;
                            fltHorgrad = fltKahant;
                        }
                    }

                    tenHorgrad[intIndex] = fltHorgrad;
                } }
            """,
                    {
                        "tenIn": tenIn,
                        "tenVer": tenVer,
                        "tenHor": tenHor,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenVergrad": tenVergrad,
                        "tenHorgrad": tenHorgrad,
                    },
                )
            )(
                grid=tuple([int((tenHorgrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenHorgrad.nelement()),
                    tenIn.data_ptr(),
                    tenVer.data_ptr(),
                    tenHor.data_ptr(),
                    tenOutgrad.data_ptr(),
                    None,
                    None,
                    tenHorgrad.data_ptr(),
                ],
            )
        # end

        return tenIngrad, tenVergrad, tenHorgrad

    # end


# end


import torch

import math
import numpy
import os
import PIL
import PIL.Image
import sys
import typing

##########################################################

assert (
    int(str("").join(torch.__version__.split(".")[0:2])) >= 13
)  # requires at least pytorch version 1.3.0

torch.set_grad_enabled(
    False
)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = (
    True  # make sure to use cudnn for computational performance
)

##########################################################

##########################################################


class Basic(torch.nn.Module):
    def __init__(
        self,
        strType: str,
        intChans: typing.List[int],
        objScratch: typing.Optional[typing.Dict] = None,
    ):
        super().__init__()

        self.strType = strType
        self.netEvenize = None
        self.netMain = None
        self.netShortcut = None

        intIn = intChans[0]
        intOut = intChans[-1]
        netMain = []
        intChans = intChans.copy()
        fltStride = 1.0

        for intPart, strPart in enumerate(self.strType.split("+")[0].split("-")):
            if strPart.startswith("conv") == True:
                intKsize = 3
                intPad = 1
                strPad = "zeros"

                if "(" in strPart:
                    intKsize = int(strPart.split("(")[1].split(")")[0].split(",")[0])
                    intPad = int(math.floor(0.5 * (intKsize - 1)))

                    if "replpad" in strPart.split("(")[1].split(")")[0].split(","):
                        strPad = "replicate"
                    if "reflpad" in strPart.split("(")[1].split(")")[0].split(","):
                        strPad = "reflect"
                # end

                if "nopad" in self.strType.split("+"):
                    intPad = 0
                # end

                netMain += [
                    torch.nn.Conv2d(
                        in_channels=intChans[0],
                        out_channels=intChans[1],
                        kernel_size=intKsize,
                        stride=1,
                        padding=intPad,
                        padding_mode=strPad,
                        bias="nobias" not in self.strType.split("+"),
                    )
                ]
                intChans = intChans[1:]
                fltStride *= 1.0

            elif strPart.startswith("sconv") == True:
                intKsize = 3
                intPad = 1
                strPad = "zeros"

                if "(" in strPart:
                    intKsize = int(strPart.split("(")[1].split(")")[0].split(",")[0])
                    intPad = int(math.floor(0.5 * (intKsize - 1)))

                    if "replpad" in strPart.split("(")[1].split(")")[0].split(","):
                        strPad = "replicate"
                    if "reflpad" in strPart.split("(")[1].split(")")[0].split(","):
                        strPad = "reflect"
                # end

                if "nopad" in self.strType.split("+"):
                    intPad = 0
                # end

                netMain += [
                    torch.nn.Conv2d(
                        in_channels=intChans[0],
                        out_channels=intChans[1],
                        kernel_size=intKsize,
                        stride=2,
                        padding=intPad,
                        padding_mode=strPad,
                        bias="nobias" not in self.strType.split("+"),
                    )
                ]
                intChans = intChans[1:]
                fltStride *= 2.0

            elif strPart.startswith("up") == True:

                class Up(torch.nn.Module):
                    def __init__(self, strType):
                        super().__init__()

                        self.strType = strType

                    # end

                    def forward(self, tenIn: torch.Tensor) -> torch.Tensor:
                        if self.strType == "nearest":
                            return torch.nn.functional.interpolate(
                                input=tenIn,
                                scale_factor=2.0,
                                mode="nearest",
                                align_corners=False,
                            )

                        elif self.strType == "bilinear":
                            return torch.nn.functional.interpolate(
                                input=tenIn,
                                scale_factor=2.0,
                                mode="bilinear",
                                align_corners=False,
                            )

                        elif self.strType == "pyramid":
                            return pyramid(tenIn, None, "up")

                        elif self.strType == "shuffle":
                            return torch.nn.functional.pixel_shuffle(
                                tenIn, upscale_factor=2
                            )  # https://github.com/pytorch/pytorch/issues/62854

                        # end

                        assert False  # to make torchscript happy

                    # end

                # end

                strType = "bilinear"

                if "(" in strPart:
                    if "nearest" in strPart.split("(")[1].split(")")[0].split(","):
                        strType = "nearest"
                    if "pyramid" in strPart.split("(")[1].split(")")[0].split(","):
                        strType = "pyramid"
                    if "shuffle" in strPart.split("(")[1].split(")")[0].split(","):
                        strType = "shuffle"
                # end

                netMain += [Up(strType)]
                fltStride *= 0.5

            elif strPart.startswith("prelu") == True:
                netMain += [
                    torch.nn.PReLU(
                        num_parameters=1,
                        init=float(strPart.split("(")[1].split(")")[0].split(",")[0]),
                    )
                ]
                fltStride *= 1.0

            elif True:
                assert False

            # end
        # end

        self.netMain = torch.nn.Sequential(*netMain)

        for strPart in self.strType.split("+")[1:]:
            if strPart.startswith("skip") == True:
                if intIn == intOut and fltStride == 1.0:
                    self.netShortcut = torch.nn.Identity()

                elif intIn != intOut and fltStride == 1.0:
                    self.netShortcut = torch.nn.Conv2d(
                        in_channels=intIn,
                        out_channels=intOut,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias="nobias" not in self.strType.split("+"),
                    )

                elif intIn == intOut and fltStride != 1.0:

                    class Down(torch.nn.Module):
                        def __init__(self, fltScale):
                            super().__init__()

                            self.fltScale = fltScale

                        # end

                        def forward(self, tenIn: torch.Tensor) -> torch.Tensor:
                            return torch.nn.functional.interpolate(
                                input=tenIn,
                                scale_factor=self.fltScale,
                                mode="bilinear",
                                align_corners=False,
                            )

                        # end

                    # end

                    self.netShortcut = Down(1.0 / fltStride)

                elif intIn != intOut and fltStride != 1.0:

                    class Down(torch.nn.Module):
                        def __init__(self, fltScale):
                            super().__init__()

                            self.fltScale = fltScale

                        # end

                        def forward(self, tenIn: torch.Tensor) -> torch.Tensor:
                            return torch.nn.functional.interpolate(
                                input=tenIn,
                                scale_factor=self.fltScale,
                                mode="bilinear",
                                align_corners=False,
                            )

                        # end

                    # end

                    self.netShortcut = torch.nn.Sequential(
                        Down(1.0 / fltStride),
                        torch.nn.Conv2d(
                            in_channels=intIn,
                            out_channels=intOut,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias="nobias" not in self.strType.split("+"),
                        ),
                    )

                # end

            elif strPart.startswith("...") == True:
                pass

            # end
        # end

        assert len(intChans) == 1

    # end

    def forward(self, tenIn: torch.Tensor) -> torch.Tensor:
        if self.netEvenize is not None:
            tenIn = self.netEvenize(tenIn)
        # end

        tenOut = self.netMain(tenIn)

        if self.netShortcut is not None:
            tenOut = tenOut + self.netShortcut(tenIn)
        # end

        return tenOut

    # end


# end


class Encode(torch.nn.Module):
    objScratch: typing.Dict[str, typing.List[int]] = None

    def __init__(
        self,
        intIns: typing.List[int],
        intOuts: typing.List[int],
        strHor: str,
        strVer: str,
        objScratch: typing.Dict[str, typing.List[int]],
    ):
        super().__init__()

        assert len(intIns) == len(intOuts)
        assert len(intOuts) == len(intIns)

        self.intRows = len(intIns) and len(intOuts)
        self.intIns = intIns.copy()
        self.intOuts = intOuts.copy()
        self.strHor = strHor
        self.strVer = strVer
        self.objScratch = objScratch

        self.netHor = torch.nn.ModuleList()
        self.netVer = torch.nn.ModuleList()

        for intRow in range(self.intRows):
            netHor = torch.nn.Identity()
            netVer = torch.nn.Identity()

            if self.intOuts[intRow] != 0:
                if self.intIns[intRow] != 0:
                    netHor = Basic(
                        self.strHor,
                        [
                            self.intIns[intRow],
                            self.intOuts[intRow],
                            self.intOuts[intRow],
                        ],
                        objScratch,
                    )
                # end

                if intRow != 0:
                    netVer = Basic(
                        self.strVer,
                        [
                            self.intOuts[intRow - 1],
                            self.intOuts[intRow],
                            self.intOuts[intRow],
                        ],
                        objScratch,
                    )
                # end
            # end

            self.netHor.append(netHor)
            self.netVer.append(netVer)
        # end

    # end

    def forward(self, tenIns: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
        intRow = 0
        for netHor in self.netHor:
            if self.intOuts[intRow] != 0:
                if self.intIns[intRow] != 0:
                    tenIns[intRow] = netHor(tenIns[intRow])
                # end
            # end
            intRow += 1
        # end

        intRow = 0
        for netVer in self.netVer:
            if self.intOuts[intRow] != 0:
                if intRow != 0:
                    tenIns[intRow] = tenIns[intRow] + netVer(tenIns[intRow - 1])
                # end
            # end
            intRow += 1
        # end

        for intRow, tenIn in enumerate(tenIns):
            self.objScratch["levelshape" + str(intRow)] = tenIn.shape
        # end

        return tenIns

    # end


# end


class Decode(torch.nn.Module):
    objScratch: typing.Dict[str, typing.List[int]] = None

    def __init__(
        self,
        intIns: typing.List[int],
        intOuts: typing.List[int],
        strHor: str,
        strVer: str,
        objScratch: typing.Dict[str, typing.List[int]],
    ):
        super().__init__()

        assert len(intIns) == len(intOuts)
        assert len(intOuts) == len(intIns)

        self.intRows = len(intIns) and len(intOuts)
        self.intIns = intIns.copy()
        self.intOuts = intOuts.copy()
        self.strHor = strHor
        self.strVer = strVer
        self.objScratch = objScratch

        self.netHor = torch.nn.ModuleList()
        self.netVer = torch.nn.ModuleList()

        for intRow in range(self.intRows - 1, -1, -1):
            netHor = torch.nn.Identity()
            netVer = torch.nn.Identity()

            if self.intOuts[intRow] != 0:
                if self.intIns[intRow] != 0:
                    netHor = Basic(
                        self.strHor,
                        [
                            self.intIns[intRow],
                            self.intOuts[intRow],
                            self.intOuts[intRow],
                        ],
                        objScratch,
                    )
                # end

                if intRow != self.intRows - 1:
                    netVer = Basic(
                        self.strVer,
                        [
                            self.intOuts[intRow + 1],
                            self.intOuts[intRow],
                            self.intOuts[intRow],
                        ],
                        objScratch,
                    )
                # end
            # end

            self.netHor.append(netHor)
            self.netVer.append(netVer)
        # end

    # end

    def forward(self, tenIns: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
        intRow = self.intRows - 1
        for netHor in self.netHor:
            if self.intOuts[intRow] != 0:
                if self.intIns[intRow] != 0:
                    tenIns[intRow] = netHor(tenIns[intRow])
                # end
            # end
            intRow -= 1
        # end

        intRow = self.intRows - 1
        for netVer in self.netVer:
            if self.intOuts[intRow] != 0:
                if intRow != self.intRows - 1:
                    tenVer = netVer(tenIns[intRow + 1])

                    if "levelshape" + str(intRow) in self.objScratch:
                        if (
                            tenVer.shape[2]
                            == self.objScratch["levelshape" + str(intRow)][2] + 1
                        ):
                            tenVer = torch.nn.functional.pad(
                                input=tenVer,
                                pad=[0, 0, 0, -1],
                                mode="constant",
                                value=0.0,
                            )
                        if (
                            tenVer.shape[3]
                            == self.objScratch["levelshape" + str(intRow)][3] + 1
                        ):
                            tenVer = torch.nn.functional.pad(
                                input=tenVer,
                                pad=[0, -1, 0, 0],
                                mode="constant",
                                value=0.0,
                            )
                    # end

                    tenIns[intRow] = tenIns[intRow] + tenVer
                # end
            # end
            intRow -= 1
        # end

        return tenIns

    # end


# end

##########################################################


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.intEncdec = [1, 1]
        self.intChannels = [32, 64, 128, 256, 512]

        self.objScratch = {}

        self.netInput = torch.nn.Conv2d(
            in_channels=3,
            out_channels=int(round(0.5 * self.intChannels[0])),
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
        )

        self.netEncode = torch.nn.Sequential(
            *(
                [
                    Encode(
                        [0] * len(self.intChannels),
                        self.intChannels,
                        "prelu(0.25)-conv(3)-prelu(0.25)-conv(3)+skip",
                        "prelu(0.25)-sconv(3)-prelu(0.25)-conv(3)",
                        self.objScratch,
                    )
                ]
                + [
                    Encode(
                        self.intChannels,
                        self.intChannels,
                        "prelu(0.25)-conv(3)-prelu(0.25)-conv(3)+skip",
                        "prelu(0.25)-sconv(3)-prelu(0.25)-conv(3)",
                        self.objScratch,
                    )
                    for intEncdec in range(1, self.intEncdec[0])
                ]
            )
        )

        self.netDecode = torch.nn.Sequential(
            *(
                [
                    Decode(
                        [0] + self.intChannels[1:],
                        [0] + self.intChannels[1:],
                        "prelu(0.25)-conv(3)-prelu(0.25)-conv(3)+skip",
                        "prelu(0.25)-up(bilinear)-conv(3)-prelu(0.25)-conv(3)",
                        self.objScratch,
                    )
                    for intEncdec in range(0, self.intEncdec[1])
                ]
            )
        )

        self.netVerone = Basic(
            "up(bilinear)-conv(3)-prelu(0.25)-conv(3)",
            [self.intChannels[1], self.intChannels[1], 51],
        )
        self.netVertwo = Basic(
            "up(bilinear)-conv(3)-prelu(0.25)-conv(3)",
            [self.intChannels[1], self.intChannels[1], 51],
        )
        self.netHorone = Basic(
            "up(bilinear)-conv(3)-prelu(0.25)-conv(3)",
            [self.intChannels[1], self.intChannels[1], 51],
        )
        self.netHortwo = Basic(
            "up(bilinear)-conv(3)-prelu(0.25)-conv(3)",
            [self.intChannels[1], self.intChannels[1], 51],
        )

        # self.load_state_dict(torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/resepconv/network-' + arguments_strModel + '.pytorch', file_name='resepconv-' + arguments_strModel))

    # end

    def forward(self, x1, x2):
        # padding if needed
        intWidth = x1.shape[3]
        intHeight = x1.shape[2]

        intPadr = (2 - (intWidth % 2)) % 2
        intPadb = (2 - (intHeight % 2)) % 2

        tenOne = torch.nn.functional.pad(
            input=x1, pad=[0, intPadr, 0, intPadb], mode="replicate"
        )
        tenTwo = torch.nn.functional.pad(
            input=x2, pad=[0, intPadr, 0, intPadb], mode="replicate"
        )
        ####

        tenSeq = [tenOne, tenTwo]

        with torch.set_grad_enabled(False):
            tenStack = torch.stack(tenSeq, 1)
            tenMean = (
                tenStack.view(tenStack.shape[0], -1)
                .mean(1, True)
                .view(tenStack.shape[0], 1, 1, 1)
            )
            tenStd = (
                tenStack.view(tenStack.shape[0], -1)
                .std(1, True)
                .view(tenStack.shape[0], 1, 1, 1)
            )
            tenSeq = [
                (tenFrame - tenMean) / (tenStd + 0.0000001) for tenFrame in tenSeq
            ]
            tenSeq = [tenFrame.detach() for tenFrame in tenSeq]
        # end

        tenOut = self.netDecode(
            self.netEncode(
                [torch.cat([self.netInput(tenSeq[0]), self.netInput(tenSeq[1])], 1)]
                + ([0.0] * (len(self.intChannels) - 1))
            )
        )[1]

        tenOne = torch.nn.functional.pad(
            input=tenOne,
            pad=[
                int(math.floor(0.5 * 51)),
                int(math.floor(0.5 * 51)),
                int(math.floor(0.5 * 51)),
                int(math.floor(0.5 * 51)),
            ],
            mode="replicate",
        )
        tenTwo = torch.nn.functional.pad(
            input=tenTwo,
            pad=[
                int(math.floor(0.5 * 51)),
                int(math.floor(0.5 * 51)),
                int(math.floor(0.5 * 51)),
                int(math.floor(0.5 * 51)),
            ],
            mode="replicate",
        )

        tenOne = torch.cat(
            [
                tenOne,
                tenOne.new_ones([tenOne.shape[0], 1, tenOne.shape[2], tenOne.shape[3]]),
            ],
            1,
        ).detach()
        tenTwo = torch.cat(
            [
                tenTwo,
                tenTwo.new_ones([tenTwo.shape[0], 1, tenTwo.shape[2], tenTwo.shape[3]]),
            ],
            1,
        ).detach()

        tenVerone = self.netVerone(tenOut)
        tenVertwo = self.netVertwo(tenOut)
        tenHorone = self.netHorone(tenOut)
        tenHortwo = self.netHortwo(tenOut)

        tenOut = sepconv_func.apply(tenOne, tenVerone, tenHorone) + sepconv_func.apply(
            tenTwo, tenVertwo, tenHortwo
        )

        tenNormalize = tenOut[:, -1:, :, :]
        tenNormalize[tenNormalize.abs() < 0.01] = 1.0
        tenOut = tenOut[:, :-1, :, :] / tenNormalize

        # crop if needed
        return tenOut[:, :, :intHeight, :intWidth]

    # end


# end

netNetwork = None

##########################################################


def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert tenOne.shape[1] == tenTwo.shape[1]
    assert tenOne.shape[2] == tenTwo.shape[2]

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert (
        intWidth <= 1280
    )  # while our approach works with larger images, we do not recommend it unless you are aware of the implications
    assert (
        intHeight <= 720
    )  # while our approach works with larger images, we do not recommend it unless you are aware of the implications

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPadr = (2 - (intWidth % 2)) % 2
    intPadb = (2 - (intHeight % 2)) % 2

    tenPreprocessedOne = torch.nn.functional.pad(
        input=tenPreprocessedOne, pad=[0, intPadr, 0, intPadb], mode="replicate"
    )
    tenPreprocessedTwo = torch.nn.functional.pad(
        input=tenPreprocessedTwo, pad=[0, intPadr, 0, intPadb], mode="replicate"
    )

    return netNetwork([tenPreprocessedOne, tenPreprocessedTwo])[
        0, :, :intHeight, :intWidth
    ].cpu()


# end
