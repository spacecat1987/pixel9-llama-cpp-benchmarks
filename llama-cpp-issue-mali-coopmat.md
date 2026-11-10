**Title:** Vulkan: KHR_coopmat makes prompt processing ~8× slower on Mali-G715 (Pixel 9 Pro XL)

**Summary**

On a Mali-G715 (Google Tensor G4, Pixel 9 Pro XL), the Vulkan backend detects `matrix cores: KHR_coopmat`. With it enabled, prompt processing is about 8× slower than with it disabled, and even slower than single-token generation. Setting `GGML_VK_DISABLE_COOPMAT=1` fixes it.

**Environment**

- Device: Google Pixel 9 Pro XL (Tensor G4), 16 GB LPDDR5X
- GPU: Mali-G715 (`uma: 1 | fp16: 1 | bf16: 0 | warp size: 16 | shared memory: 32768 | int dot: 1`)
- OS: Android, Termux (no root)
- Vulkan instance version: 1.4.368; device apiVersion 1.4.343
- Driver: ARM proprietary (`DRIVER_ID_ARM_PROPRIETARY`), driverVersion 54.3.0, driverInfo `v1.r54p3-00eac0.03d8d836cbf5c9f29d765e58a6bdfb98`
- Vendor ID 0x13b5, device ID 0xb8a20000, conformance 1.4.5.0
- llama.cpp: build 5851a8e50 (12116), built with `-DGGML_VULKAN=ON`, Clang 21.1.8

**Reproduce**

```bash
# default (coopmat on)
./build/bin/llama-bench -m Llama-3.2-1B-Instruct-Q4_0.gguf -p 512 -n 128 -t 3 -ngl 99 -r 3
# coopmat off
GGML_VK_DISABLE_COOPMAT=1 ./build/bin/llama-bench -m Llama-3.2-1B-Instruct-Q4_0.gguf -p 512 -n 128 -t 3 -ngl 99 -r 3
```

**Results (Llama 3.2 1B Q4_0, t/s)**

| | pp512 | tg128 |
|---|---|---|
| matrix cores: KHR_coopmat (default) | 19.27 ± 0.19 | 21.57 ± 0.03 |
| matrix cores: none (`GGML_VK_DISABLE_COOPMAT=1`) | 155.03 ± 0.44 | 21.63 ± 0.05 |

The same pattern appeared on Llama 3.2 3B Q4_0 (pp512 6.62 t/s with coopmat on).

**Suggestion**

Consider disabling the coopmat path by default on Mali (or this driver), since the non-coopmat path is much faster for batched matmul here.
