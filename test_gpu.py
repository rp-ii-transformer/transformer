import cupy
print("GPU disponível:", cupy.cuda.is_available())

ver = cupy.cuda.runtime.runtimeGetVersion()
major = ver // 1000
minor = (ver % 1000) // 10
print(f"CuPy rodando sobre CUDA {major}.{minor}")
