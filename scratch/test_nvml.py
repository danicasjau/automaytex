import ctypes

def get_vram_info():
    try:
        nvml = ctypes.CDLL(r"C:\Windows\System32\nvml.dll")
        nvml.nvmlInit_v2()
        
        handle = ctypes.c_void_p()
        nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
        
        class struct_c_nvmlMemory(ctypes.Structure):
            _fields_ = [
                ('total', ctypes.c_ulonglong),
                ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong),
            ]
            
        mem = struct_c_nvmlMemory()
        nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
        
        nvml.nvmlShutdown()
        
        return mem.used, mem.total
    except Exception as e:
        print(e)
        return 0, 0

used, total = get_vram_info()
print(f"Used: {used / (1024**2)} MB, Total: {total / (1024**2)} MB")
