import taichi as ti
import taichi.math as tm

@ti.kernel
def costvol_out(tenOne: ti.types.ndarray(), tltOne: ti.types.ndarray(), tenTwo: ti.types.ndarray(), tenOut: ti.types.ndarray()):
    N, C, H, W = tenOut.shape
    for i, ch, y, x in ti.ndrange(N, C, H, W):
        for intValue in range(tenOne.shape[1]):
            tltOne[intValue] = tenOne[i, intValue, y, x]

        tenOut_ch = 0
        for intOy in range(y - 4, y + 4 + 1):
            for intOx in range(x - 4, x + 4 + 1):
                point = tm.ivec2(intOx, intOy)
                fltValue = 0.0
                for intValue in range(ch):
                    if (point.y >= 0) and (point.y < H) and (point.x >= 0) and (point.x < W):
                        fltValue += ti.abs(tltOne[intValue] - tenTwo[i, intValue, point.y, point.x])
                    else:
                        fltValue += ti.abs(tltOne[intValue])
                    
                """
                Ig it accesses the first value in channels
                int intOffset = OFFSET_4(tenOut, intN, 0, intY, intX);
                tenOut[intOffset] = fltValue / SIZE_1(tenOne);
                intOffset += SIZE_2(tenOut) * SIZE_3(tenOut);
                """
                tenOut[i, tenOut_ch, y, x] = fltValue / tenOne.shape[1]
                tenOut_ch += 1

def worker_interface(op_name, tensors):
    if op_name == "costvol_out":
        tenOne, tenTwo = tensors
        tenOut = tenOne.new_zeros([tenOne.shape[0], 81, tenOne.shape[2], tenOne.shape[3]])
        tltOne = tenOne.new_zeros([tenOne.shape[1]])
        costvol_out(tenOne, tltOne, tenTwo, tenOut)
        return (tenOut, )
    
    raise NotImplementedError(op_name)
