# llama.cpp on a Pixel 9 Pro XL (Tensor G4): CPU, Mali GPU and concurrent benchmarks

Benchmarks of local LLM inference on a Google Pixel 9 Pro XL using llama.cpp inside Termux (no root). The aim is to find out how fast and how efficiently a flagship phone can run small language models, and which settings actually matter.

**Key findings**

1. **Token generation is memory-bandwidth bound.** On the CPU, effective bandwidth stays at roughly **22–25 GB/s regardless of model size**, so generation speed ≈ 22 ÷ model file size in GB.
2. **The Cortex-X4 prime core adds nothing to generation.** Pinning to the three Cortex-A720 cores gives the same tokens/sec with less heat and much more stable results. The X4 does help prompt processing (~25%) when the phone is cool.
3. **Thermal throttling starts within a couple of minutes** at full load and has a bigger effect than thread count. Test order matters; whichever configuration runs first looks better.
4. **Disabling Vulkan cooperative matrix ("matrix cores") on the Mali-G715 makes GPU prompt processing ~8× faster** (19 → 155 tok/s on Llama 3.2 1B).
5. **CPU and GPU running separate models at the same time reach ~32 GB/s combined**, about **45% more total throughput** than the CPU alone.

---

## Device and software

| Item | Value |
|---|---|
| Phone | Google Pixel 9 Pro XL, 16 GB LPDDR5X |
| SoC | Google Tensor G4: 1× Cortex-X4 (3.1 GHz), 3× Cortex-A720 (2.6 GHz), 4× Cortex-A520 (1.95 GHz) |
| GPU | Arm Mali-G715 (Vulkan) |
| Environment | Termux (F-Droid build), no root |
| Compiler | Clang 21.1.8 |
| Vulkan loader | 1.4.367 |
| llama.cpp | build 5851a8e50 (12116), November 2026 |

CPU numbering on Tensor G4 (from `cpuinfo_max_freq`):

| CPU | Core | Max clock |
|---|---|---|
| cpu0–3 | Cortex-A520 | 1.95 GHz |
| cpu4–6 | Cortex-A720 | 2.6 GHz |
| cpu7 | Cortex-X4 | 3.1 GHz |

---

## Build instructions (Termux)

```bash
pkg update && pkg upgrade
pkg install git cmake clang shaderc vulkan-headers vulkan-loader-android spirv-headers util-linux
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_VULKAN=ON \
  -DCMAKE_C_FLAGS="-march=armv8.6-a+dotprod+i8mm+fp16" \
  -DCMAKE_CXX_FLAGS="-march=armv8.6-a+dotprod+i8mm+fp16"

termux-wake-lock
cmake --build build -j 4
```

Notes:
- `spirv-headers` is required for the Vulkan backend; without it CMake fails with *Could not find SPIRV-Headers*.
- Without `+fp16`, CMake reports `HAVE_FP16_VECTOR_ARITHMETIC - Failed`.
- Building with `-j 8` was killed by Android (`signal 9`) during Vulkan shader generation. `-j 4` worked. Enabling **Developer options → Disable child process restrictions** also helps.
- Keep Termux in the foreground during builds and benchmarks; in the background Android moves it to the slow cores.

### Running CPU-only in a Vulkan build

In a Vulkan build, `-ngl 0` alone is **not** a clean CPU test: llama.cpp still offloads large prompt-processing operations to the GPU, copying weights back and forth. Prompt processing on a 3B model dropped to **6.3 tok/s** this way. Hide the GPU completely instead:

```bash
GGML_VK_VISIBLE_DEVICES= ./build/bin/llama-bench -m MODEL.gguf -p 512 -n 128 -t 3 -ngl 0
```

(The `backend` column still says `Vulkan` because of how the binary was built; `Found 0 Vulkan devices` confirms the GPU is not used.)

### Models

All from Hugging Face (bartowski GGUF repositories):
- Llama 3.2 1B Instruct Q4_0 (729.75 MiB)
- Qwen 2.5 1.5B Instruct Q4_0 (888.43 MiB)
- Llama 3.2 3B Instruct Q4_0 (1.78 GiB)
- Llama 3.1 8B Instruct IQ4_NL (4.35 GiB)

The older `Q4_0_4_4`, `Q4_0_4_8` and `Q4_0_8_8` files are no longer supported by current llama.cpp; it repacks plain `Q4_0` (and `IQ4_NL`) for ARM at load time instead. Use those.

Tip for finding exact filenames:
```bash
curl -s https://huggingface.co/api/models/bartowski/REPO-NAME | grep -o '"rfilename":"[^"]*Q4[^"]*"'
```

---

## Results

All numbers are tokens/second from `llama-bench`, mean ± standard deviation. `pp512` = prompt processing of 512 tokens; `tg128` = generating 128 tokens.

### 1. CPU vs GPU, default settings (Llama 3.2 3B Q4_0, 4 threads, unpinned)

| Setup | pp512 | tg128 |
|---|---|---|
| CPU (GPU hidden) | 64.22 ± 11.67 | 9.74 ± 1.39 |
| GPU, `-ngl 99` (matrix cores on) | 6.62 ± 0.08 | 8.64 ± 0.01 |
| Vulkan build, `-ngl 0`, GPU visible | 6.33 ± 0.03 | – |

### 2. Thread sweep (Llama 3.2 3B Q4_0, unpinned, run in order 1→6)

| Threads | pp512 | tg128 |
|---|---|---|
| 1 | 22.47 ± 4.41 | 6.30 ± 0.47 |
| 2 | 42.45 ± 5.49 | 9.95 ± 0.32 |
| 3 | 61.24 ± 0.10 | 11.98 ± 0.52 |
| 4 | 53.36 ± 11.01 | 8.37 ± 0.01 |
| 5 | 44.90 ± 0.08 | 8.28 ± 0.07 |
| 6 | 47.87 ± 1.40 | 8.02 ± 0.15 |

Repeating with the order reversed (`-t 4,3`, first run of the day) flipped the result: 4 threads gave 53.64 / 10.71, then 3 threads gave 48.49 / 7.22. **The drop after 3 threads in the sweep is largely thermal, not a thread-count effect.** Whichever configuration runs first gets a cooler phone.

### 3. Core pinning: A720 only vs A720 + X4 (Llama 3.2 3B Q4_0)

```bash
# A: three A720 cores
GGML_VK_VISIBLE_DEVICES= taskset -c 4-6 ./build/bin/llama-bench -m MODEL.gguf -p 512 -n 128 -t 3 -ngl 0
# B: three A720 cores + X4
GGML_VK_VISIBLE_DEVICES= taskset -c 4-7 ./build/bin/llama-bench -m MODEL.gguf -p 512 -n 128 -t 4 -ngl 0
```

| Session | Run | pp512 | tg128 |
|---|---|---|---|
| 1 | A first | 61.53 ± 1.50 | 13.09 ± 0.28 |
| 1 | B second | 63.24 ± 16.12 | 11.05 ± 1.37 |
| 2 (after cooldown) | B first | 75.93 ± 7.17 | 12.42 ± 0.51 |
| 2 | A second | 60.62 ± 0.21 | 12.65 ± 0.64 |

- **Generation:** A and B tie (~12.4–13.1 tok/s). The X4 adds no generation speed.
- **Prompt processing:** the X4 adds ~25% when cool, but results are noisy as it throttles within the run.
- **Stability:** the A720-only configuration gave almost identical results in both sessions, whether run first or second.

**Recommendation for sustained generation (e.g. a worker phone): pin to cpu4–6 with 3 threads.** For long prompts, llama.cpp's separate `-tb` (batch threads) option can add the X4 for the prompt phase only.

### 4. Model size vs speed (CPU, A720 pinned, 3 threads)

| Model | File size | pp512 | tg128 | Effective bandwidth* |
|---|---|---|---|---|
| Llama 3.2 1B Q4_0 | 0.77 GB | 168.52 ± 0.43 | 28.63 ± 0.41 | ~21.9 GB/s |
| Qwen 2.5 1.5B Q4_0 | 0.93 GB | 124.03 ± 3.00 | 23.89 ± 0.34 | ~22.3 GB/s |
| Llama 3.2 3B Q4_0 | 1.91 GB | ~61 | 12.65–13.09 | ~24–25 GB/s |
| Llama 3.1 8B IQ4_NL (`-r 3`) | 4.67 GB | 15.52 ± 0.76 | 4.69 ± 0.86 | ~21.9 GB/s |

\* File size × generation speed. Each generated token reads (almost) all weights once, so this approximates the memory bandwidth actually achieved.

The CPU achieves a near-constant ~22 GB/s, which looks like the practical ceiling for the CPU cluster's access to memory on this SoC. That makes generation speed predictable from file size alone.

IQ4_NL prompt processing (15.5 tok/s for 8B) was lower than predicted from the Q4_0 results, probably because it uses a different kernel path from Q4_0's i8mm-accelerated one.

### 5. Mali GPU: cooperative matrix on vs off (Llama 3.2 1B Q4_0, `-ngl 99`)

```bash
GGML_VK_DISABLE_COOPMAT=1 taskset -c 4-6 ./build/bin/llama-bench -m MODEL.gguf -p 512 -n 128 -t 3 -ngl 99 -r 3
```

| Setup | pp512 | tg128 |
|---|---|---|
| CPU (A720 ×3), for reference | 168.52 ± 0.43 | 28.63 ± 0.41 |
| GPU, matrix cores **on** (default, `KHR_coopmat`) | 19.27 ± 0.19 | 21.57 ± 0.03 |
| GPU, matrix cores **off** | **155.03 ± 0.44** | 21.63 ± 0.05 |

With cooperative matrix enabled, GPU prompt processing was *slower than single-token generation*, which should not happen. Disabling it made prompt processing **8× faster**, close to the CPU. Generation is unaffected (~75% of CPU speed, ~16.5 GB/s effective) and extremely stable.

To make it permanent in Termux:
```bash
echo 'export GGML_VK_DISABLE_COOPMAT=1' >> ~/.bashrc
```

### 6. CPU and GPU at the same time (Llama 3.2 1B Q4_0, generation only)

Two Termux sessions started within ~2 seconds of each other:

```bash
# Session 1: CPU on A720 cores
GGML_VK_VISIBLE_DEVICES= taskset -c 4-6 ./build/bin/llama-bench -m MODEL.gguf -p 0 -n 512 -t 3 -ngl 0
# Session 2: GPU, its host thread on the X4 core
taskset -c 7 ./build/bin/llama-bench -m MODEL.gguf -p 0 -n 512 -t 1 -ngl 99
```

| | Alone | Simultaneously |
|---|---|---|
| CPU | 28.63 | 21.18 ± 1.89 |
| GPU | 21.57 | 20.34 ± 0.24 |
| **Total** | 28.63 (best single) | **41.52** |

Combined effective bandwidth ≈ **32 GB/s**, above the CPU's ~22 GB/s ceiling, so the GPU has its own route to memory. The CPU absorbed most of the contention (−26%); the GPU barely slowed.

**Practical meaning:** to run two models on one phone, put one on the CPU and one on the GPU. This does **not** speed up a single answer (splitting one model between CPU and GPU processes layers in sequence); it raises total throughput for parallel jobs.

---

## Limitations

- One device, one build. Results will differ with other llama.cpp versions, Android versions and drivers.
- No power measurements yet, so energy per token is not measured.
- Phone temperature and room temperature were not logged; cooldown was judged by touch.
- 3–5 repetitions per test.
- Bandwidth figures are approximations derived from file size × generation speed.

## Open questions / ideas for contributions

- **Power and energy per token** for each configuration (e.g. via `termux-battery-status` current readings, or an external USB power meter).
- **Speculative decoding**: Llama 3.1 8B with Llama 3.2 1B as draft (the draft is ~6× faster than the target here).
- **Snapdragon comparison** (e.g. Snapdragon 8 Elite / 8 Elite Gen 5 with Adreno and llama.cpp's OpenCL backend) using the same commands.
- **Fan or passive cooling** vs none, to quantify sustained-performance gains.
- Whether the cooperative-matrix slowdown affects other Mali GPUs.
- Gemma 3n and other mobile-optimised models.

If you run these tests on another phone, please open an issue or pull request with your results table, device, SoC and llama.cpp build number.

## Licence

Data and text: CC BY 4.0. Commands and scripts: MIT.
