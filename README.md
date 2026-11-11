# llama.cpp on a Pixel 9 Pro XL (Tensor G4): CPU, Mali GPU and concurrent benchmarks

Benchmarks of local LLM inference on a Google Pixel 9 Pro XL using llama.cpp inside Termux (no root). The aim is to find out how fast and how efficiently a flagship phone can run small language models, and which settings actually matter.

**Key findings**

1. **Token generation is memory-bandwidth bound.** On the CPU, effective bandwidth stays at roughly **22–25 GB/s regardless of model size**, so generation speed ≈ 22 ÷ model file size in GB.
2. **The Cortex-X4 prime core adds nothing to generation.** Pinning to the three Cortex-A720 cores gives the same tokens/sec with less heat and much more stable results. The X4 does help prompt processing (~25%) when the phone is cool.
3. **Thermal throttling starts within a couple of minutes** at full load and has a bigger effect than thread count. Test order matters; whichever configuration runs first looks better.
4. **Disabling Vulkan cooperative matrix ("matrix cores") on the Mali-G715 makes GPU prompt processing ~8× faster** (19 → 155 tok/s on Llama 3.2 1B).
5. **CPU and GPU running separate models at the same time reach ~32 GB/s combined**, about **45% more total throughput** than the CPU alone.
6. **Batching parallel requests nearly triples CPU generation throughput**: 4 simultaneous requests on Llama 3.2 1B give **78 tok/s in total** (vs 28 for one), with each request still at ~19.5 tok/s.
7. **Batched CPU + GPU together reach ~111 tok/s at ~0.11 J per token** (whole phone), about **4.3× less energy per token** than single-stream generation (0.48 J; ~4.5× counting inference power only). Power does not add up: adding the GPU to a running CPU job costs only ~0.5 W, which suggests the shared memory system is the main power cost.
8. **Speculative decoding needs short drafts on this phone.** Checking 4 tokens in one pass costs only ~1.35× generating a single token, but checking 9 (the usual draft of 8 + 1) costs ~3.2×. With a draft length of 3 and Qwen 2.5 0.5B as draft, Qwen 2.5 7B went from **6.0 to 9.2 tok/s** on code and **7.9 tok/s** on prose.
9. **Larger models throttle hard:** Qwen 2.5 7B generation fell from 6.0 to 3.8 tok/s between two back-to-back runs.
10. **Fewer bits is not faster on this CPU.** Every format below 4 bits (Q3_K_M, Q2_K, IQ3_M, IQ2_M) generated more slowly than Q4_0 on Qwen 2.5 7B, from 4.9 down to 1.6 tok/s against 6.0. These formats are compute-bound: unpacking the weights, not reading them, is the limit. For them the X4 core *does* help (+76% for Q3_K_M), so the best core setup depends on the format.
11. **Ternary BitNet b1.58 2B: a llama.cpp bug, two speed-ups, and a fair comparison.** llama.cpp runs this model with the wrong activation (SiLU instead of ReLU²), giving perplexity ~102; a one-line fix brings it to 16.9. Quantising its large output layer to Q8_0 costs no measurable quality, and ik_llama.cpp's runtime repacking (which works on ARM) lifts generation from 18.2 to **25.3 tok/s**. But at equal WikiText perplexity, **Llama 3.2 1B Q4_0 is faster on this phone** (27–30 tok/s, 1.8× faster prompts) **with about the same energy per token** (~0.3 J).

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
| ik_llama.cpp (section 11 only) | build 4e24d0b2 (5223), CPU only |

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
- Qwen 2.5 7B Instruct Q4_0 (4.13 GiB)
- Qwen 2.5 0.5B Instruct Q4_0 (draft model)
- Qwen 2.5 7B Instruct Q3_K_M, Q2_K, IQ3_M, IQ2_M (format comparison)
- BitNet b1.58 2B4T, TQ2_0 ternary (1.11 GiB, 2.41 B parameters; from `Synapticode/bitnet-b1.58-2B-4T-tq2_0-gguf`), plus derived files: Q8_0 output layer (`llama-quantize`) and PQ2_0 versions for ik_llama.cpp (section 11)

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

\* File size × generation speed. Each generated token reads (almost) all weights once, so this approximates the memory bandwidth actually achieved. Llama 3.2 1B/3B and Qwen 2.5 1.5B share one table between input embeddings and output layer, so their file size is close to the bytes read per token. Llama 3.1 8B has a separate input-embedding table (128,256 × 4,096, roughly 0.3 GB at IQ4_NL) of which only one row is read per token, so its true figure is nearer ~20.5 GB/s.

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

With cooperative matrix enabled, GPU prompt processing was *slower than single-token generation*, which should not happen. Disabling it made prompt processing **8× faster**, close to the CPU. Generation is unaffected (~75% of CPU speed, ~16.5 GB/s effective) and extremely stable. The same slowdown has been reported on a Mali-G925 (MediaTek Dimensity 9400), about 2.4× there: [llama.cpp #30006](https://github.com/ggml-org/llama.cpp/issues/30006), open as of November 2026.

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

Note: `GGML_VK_DISABLE_COOPMAT=1` would not raise the GPU figure here. Cooperative matrix only affects batched prompt processing; single-stream generation was identical with it on and off (21.57 vs 21.63 tok/s, section 5).

### 7. Batched generation: serving several requests at once (Llama 3.2 1B Q4_0, CPU, A720 pinned)

Generation is memory-bound, so the CPU has spare compute while it waits for weights. Serving several requests at once lets each read of the weights produce a token for every request.

```bash
GGML_VK_VISIBLE_DEVICES= taskset -c 4-6 ./build/bin/llama-batched-bench -m MODEL.gguf -c 4096 -npp 128 -ntg 128 -npl 1,2,4,8 -t 3
```

| Parallel requests | Prompt (S_PP) | Total generation (S_TG) | Generation per request | Overall (S) |
|---|---|---|---|---|
| 1 | 176.30 | 27.95 | 27.95 | 48.25 |
| 2 | 179.30 | 34.65 | 17.33 | 58.08 |
| **4** | 182.53 | **78.08** | **19.52** | 109.37 |
| 8 | 178.62 | 80.10 | 10.01 | 110.60 |

(t/s; each request has a 128-token prompt and generates 128 tokens.)

- **4 parallel requests is the sweet spot:** 2.8× the single-stream total, while each request still gets ~19.5 tok/s.
- **At 8 the total plateaus** (~80 tok/s): compute, not memory, is now the limit, so extra requests only split the same output.
- **2 requests gave little gain** (1.24×). A possible explanation is that llama.cpp's repacked ARM Q4_0 kernels process 4 rows at a time, so a batch of 2 falls back to a slower path. Not confirmed; if true, batch sizes in multiples of 4 matter.
- 78 tok/s × 0.77 GB ≈ 60 GB/s "equivalent", well beyond the ~22 GB/s physical ceiling, because each weight read now serves several requests.

Caveat: these are short conversations (128 + 128 tokens). Longer contexts add attention work and will reduce the gain.

**Practical meaning:** `llama-server` serves parallel users the same way (its `-np` option). One phone handling 4 requests at ~20 tok/s each is far more useful for a server or worker fleet than one request at 28 tok/s.

### 8. Power and energy per token

**Method (no root).** Termux:API's `termux-battery-status` reports battery current (µA, negative when discharging) and voltage (mV). The phone ran on battery (charger unplugged), with fixed screen brightness, Battery Saver off and Termux in the foreground. A logger sampled every ~1.5 s during each run.

Requirements: the **Termux:API app** (from the same source as Termux, e.g. F-Droid; `pkg install termux-api` alone is not enough) plus `pkg install termux-api jq`. Reading `/sys/class/power_supply/battery/current_now` directly is blocked without root.

Logger (`~/powerlog.sh`):
```bash
#!/data/data/com.termux/files/usr/bin/bash
out=${1:-power.csv}
echo "time,current_ua,voltage_mv,temp_c" > "$out"
while true; do
  termux-battery-status | jq -r '[(now|floor), .current, .voltage, .temperature] | @csv' >> "$out"
  sleep 1
done
```

Summary (`~/powersum.sh`):
```bash
#!/data/data/com.termux/files/usr/bin/bash
awk -F, 'NR>1 {c=($2<0?-$2:$2); s+=c/1e6*$3/1000; n++; t=$4} END {printf "samples: %d  avg power: %.2f W  end temp: %.1f C\n", n, s/n, t}' "$1"
```

Usage:
```bash
~/powerlog.sh idle.csv & sleep 60; kill $!; ~/powersum.sh idle.csv
~/powerlog.sh load.csv & <benchmark command>; kill $!; ~/powersum.sh load.csv
```

For CPU and GPU together, a single script starts both at the same moment and logs until both finish (`~/both.sh`):
```bash
#!/data/data/com.termux/files/usr/bin/bash
cd ~/llama.cpp
M=~/models/llama1b-q4_0.gguf
~/powerlog.sh both.csv & LOG=$!
GGML_VK_DISABLE_COOPMAT=1 taskset -c 7 ./build/bin/llama-batched-bench -m $M -c 4096 -npp 128 -ntg 280 -npl 4 -t 1 -ngl 99 > gpu_out.txt 2>&1 &
GPU=$!
GGML_VK_VISIBLE_DEVICES= taskset -c 4-6 ./build/bin/llama-batched-bench -m $M -c 4096 -npp 128 -ntg 512 -npl 4 -t 3 > cpu_out.txt 2>&1
wait $GPU
kill $LOG
echo "CPU:"; grep '^| *128' cpu_out.txt
echo "GPU:"; grep '^| *128' gpu_out.txt
~/powersum.sh both.csv
```
(The GPU job generates 280 tokens per request and the CPU job 512, so both generation phases take ~28 s and overlap fully.)

**Idle baseline** (screen on, Termux open, logger running): 3.62 W and 3.78 W, average **3.70 W**. "Inference power" below is total power minus this baseline.

**Results (Llama 3.2 1B Q4_0; CPU = 3× A720 pinned, 3 threads; GPU = Mali, coopmat off):**

| Setup | Runs | Generation | Phone power | Inference power | J/token (inference) | J/token (whole phone) |
|---|---|---|---|---|---|---|
| CPU, 1 request | 1 | 27.3 tok/s | 13.17 W | 9.47 W | 0.35 | 0.48 |
| CPU, 4 requests | 3 | 72.3 tok/s | 11.80 W | 8.10 W | 0.112 | 0.163 |
| GPU, 4 requests | 2 | 39.3 tok/s | 8.55 W | 4.85 W | 0.123 | 0.218 |
| **CPU + GPU, 4 requests each** | 3 | **111.2 tok/s** | **12.32 W** | **8.62 W** | **0.078** | **0.111** |

Individual runs: CPU batched 11.16 / 12.00 / 12.25 W (71.8 / 72.3 / 72.8 tok/s); GPU batched 8.39 / 8.71 W (38.9 / 39.7 tok/s); combined 11.98 / 12.64 / 12.33 W (110.1 / 110.5 / 113.1 tok/s).

**Findings:**
- **Batching cuts energy per token ~3×** on the CPU alone (0.48 → 0.16 J whole phone).
- **CPU and GPU are similarly efficient on their own** (~0.12 J per token of inference power each).
- **Power is strongly sub-additive.** If the two added up, both together would need ~13 W of inference power (8.10 + 4.85); measured is 8.62 W. Adding the GPU to a running CPU job cost only ~0.5 W for an extra ~39 tok/s, within run-to-run noise (~±0.4 W). Neither side slowed down, so this is not a power cap. The likely explanation is that the memory system (DRAM, memory controller, interconnect) is the dominant power cost and is already running at full clock for the CPU; the GPU then shares it almost for free.
- **Single-stream generation is expensive** (~13 W for 27 tok/s) for the same reason: most of the energy goes into moving weights, not computing.
- In batched mode the CPU and GPU did **not** slow each other down (CPU 70.9–73.9 tok/s with the GPU running vs 71.8–72.8 alone), unlike single-stream (section 6), because batched generation is compute-bound rather than bandwidth-bound.

Caveats: the battery gauge is sampled every ~1.5 s (21–46 samples per run) and identical runs varied by ~0.7 W. The figures are good for comparing setups; absolute watts should be cross-checked with a USB-C power meter.

### 9. Speculative decoding

A small draft model proposes several tokens; the large model checks them all in one batched pass and keeps the ones it agrees with. Both models ran on the CPU (A720 cores):

```bash
GGML_VK_VISIBLE_DEVICES= taskset -c 4-6 ./build/bin/llama-speculative -m TARGET.gguf -md DRAFT.gguf -p "PROMPT" -n 256 -t 3 -td 3 -c 2048 --spec-draft-n-max 3 --temp 0
```

Notes:
- `--draft-max` has been renamed `--spec-draft-n-max` in current llama.cpp.
- **Always set `-c`** (e.g. `-c 2048`). Without it, Termux was killed by Android, apparently because llama.cpp reserved memory for the 8B model's full 128K context.
- Draft and target must share a tokenizer (Llama 3.1 8B + Llama 3.2 1B; Qwen 2.5 7B + Qwen 2.5 0.5B/1.5B).

**Llama 3.1 8B IQ4_NL + Llama 3.2 1B Q4_0 draft (code prompt):**

| Draft length | Speed | Acceptance |
|---|---|---|
| none (baseline, llama-bench) | 4.69 tok/s | – |
| 4 | 6.45 tok/s | 81.1% |
| 8 | 7.01 tok/s | 83.1% |

Checking cost grew almost linearly with batch size (~76 ms per token), because IQ4_NL's batched path is slow on this CPU (prompt processing only 15.5 tok/s). That caps the achievable speed-up at roughly 9 tok/s however the draft is tuned.

**Qwen 2.5 7B Q4_0 baseline (Q4_0 has a faster batched path):**

| Run | pp512 | tg128 |
|---|---|---|
| Cold (first run of the morning) | 27.96 ± 0.13 | **6.02 ± 0.07** |
| Immediately after | 25.55 ± 1.42 | 3.82 ± 0.61 |
| Previous evening | 26.65 ± 0.80 | 4.60 ± 0.41 |

**Cost of checking different batch sizes (Qwen 2.5 7B Q4_0, `llama-bench -p 4,8,9,12,16 -n 0`):**

| Tokens checked | Speed | Time per check | vs one generated token (~0.17 s) |
|---|---|---|---|
| 4 | 17.76 tok/s | 0.23 s | 1.35× |
| 8 | 18.85 tok/s | 0.42 s | 2.5× |
| **9** | **16.92 tok/s** | **0.53 s** | **3.2×** |
| 12 | 19.96 tok/s | 0.60 s | 3.6× |
| 16 | 26.60 tok/s | 0.60 s | 3.6× |

- A batch of **9** (the default-style draft of 8 + 1) is a bad size: the 9th token costs ~0.11 s on its own. Kernels evidently work in fixed-size blocks; 12 and 16 cost the same.
- A batch of 4 costs only 1.35× a single token, which makes **short drafts** the efficient choice here.

**Qwen 2.5 7B Q4_0 speculative results:**

| Draft model | Draft length | Prompt | Speed | Acceptance |
|---|---|---|---|---|
| none (cold baseline) | – | – | 6.02 tok/s | – |
| Qwen 2.5 1.5B Q4_0 | 8 | code | 5.62 tok/s | 57.6% |
| Qwen 2.5 1.5B Q4_0 | 3 | code | 8.17 tok/s | 82.9% |
| **Qwen 2.5 0.5B Q4_0** | **3** | **code** | **9.21 tok/s** | 76.9% |
| **Qwen 2.5 0.5B Q4_0** | **3** | **prose** | **7.92 tok/s** | 50.3% |

(Code prompt: "Write a Python function that checks whether a number is prime, with comments." Prose prompt: "Explain in simple terms how ocean tides work.")

- With draft length 8 the 9-token checks made speculative decoding **slower** than no speculation. Draft length 3 (4-token checks) turned it into a 1.36× gain.
- The 0.5B draft cut drafting and overhead from ~46% to ~29% of total time, for **1.53× the cold baseline** on code. Checking is now ~71% of the time, so the target's small-batch speed is the remaining limit.
- On prose, acceptance fell to ~50%, but each round still yielded ~2.5 tokens, giving **1.32×** the cold baseline.
- With `--temp 0` the target decides every token, so quality is unaffected, but output is **not word-for-word identical** across settings: different batch sizes take slightly different numerical paths, which can tip close choices.

A simple model (per-position acceptance ~0.87–0.9, check times from the batch-size table, ~40–75 ms per drafted token) predicted ~9 tok/s for draft length 3 with the 1.5B draft and ~10–11 with the 0.5B; measured 8.2 and 9.2.

### 10. Lower-bit formats: smaller files, slower generation (Qwen 2.5 7B)

Since generation is memory-bound for Q4_0, smaller files should in principle generate faster. Each format was run from cold with at least 10 minutes' cooldown in between:

```bash
GGML_VK_VISIBLE_DEVICES= taskset -c 4-7 ./build/bin/llama-bench -m MODEL.gguf -p 4,512 -n 128 -t 4 -r 3
```

`pp4` is included because it is the batch size that matters for speculative decoding (section 9).

| Format | File size | Cores | pp4 | pp512 | tg128 | Effective bandwidth* |
|---|---|---|---|---|---|---|
| **Q4_0** | 4.43 GB | 3× A720 | **17.76 ± 0.32** | **27.96 ± 0.13** | **6.02 ± 0.07** | ~27 GB/s |
| Q3_K_M | 3.80 GB | 3× A720 | 5.60 ± 0.04 | 5.41 ± 0.90 | 2.80 ± 0.13 | ~11 GB/s |
| Q3_K_M | 3.80 GB | 3× A720 + X4 | 8.05 ± 0.74 | 8.69 ± 0.58 | 4.94 ± 0.06 | ~19 GB/s |
| Q2_K | 3.01 GB | 3× A720 + X4 | 5.26 ± 0.05 | 7.33 ± 1.34 | 3.04 ± 0.49 | ~9 GB/s |
| IQ3_M (IQ3_S mix, 3.66 bpw) | 3.56 GB | 3× A720 + X4 | 2.16 ± 0.77 | 2.01 ± 0.04 | 1.64 ± 0.11 | ~6 GB/s |
| IQ2_M (2.7 bpw) | 2.77 GB | 3× A720 + X4 | 3.80 ± 0.02 | 2.43 ± 0.26 | 1.94 ± 0.01 | ~5 GB/s |

\* File size × generation speed. The Q4_0 row is the cold baseline from section 9 (its pp4 comes from the batch-size run there). Qwen 2.5 7B has a separate input-embedding table (~0.3 GB at Q4_0) of which only one row is read per token, so file size overstates the bytes read: the Q4_0 figure is really ~25 GB/s, in line with section 4.

**Findings:**
- **Every sub-4-bit format is slower than Q4_0**, at every batch size. Ranked by speed, the order is almost the reverse of file size.
- **These formats are compute-bound.** They use only 5–19 GB/s of memory bandwidth, and prompt processing (pp512) is barely faster than generation, so there is no batching benefit. Unpacking the weights, not reading them, is the bottleneck.
- **K formats** (Q3_K, Q2_K) lose 20–50% against Q4_0. **Codebook-based IQ formats** (IQ3, IQ2), which store weights as indexes into lookup tables, lose about 70%.
- **The best core setup depends on the format.** For memory-bound Q4_0 the X4 adds nothing (section 3). For compute-bound Q3_K_M, adding the X4 raised generation by 76% (2.80 → 4.94 tok/s).
- **For speculative decoding these formats are poor targets:** their small-batch checking speed (pp4) is 2–8× slower than Q4_0's.
- Q4_0 has a repacked, i8mm-accelerated ARM path in llama.cpp; the others do not have an equivalent here. **On this CPU, fewer bits only pays off with simple, fast kernels.**

Caveats: output quality was not assessed, and the sub-4-bit formats were tested only with the X4 included (plus Q3_K_M on the A720s alone).

### 11. Ternary weights: BitNet b1.58 2B4T

Ternary models store each weight as −1, 0 or +1. BitNet b1.58 2B4T (2.41 B parameters) was **trained** this way, unlike normal models compressed to ternary afterwards. It was run here in two ways:
- **mainline llama.cpp**, TQ2_0 format (2.06 bits per weight);
- **ik_llama.cpp** (a fork by Iwan Kawrakow), PQ2_0 format with runtime repacking (`-rtr`), build 4e24d0b2 (5223), CPU only.

#### 11.1 A bug in llama.cpp's BitNet support, and a one-line fix

The first perplexity test gave **101.84 ± 5.14** on WikiText-2, and in chat the model answered "Paris" and then looped ("In the same way, Paris is the capital of France…").

**Cause:** BitNet b1.58 2B4T uses a **squared ReLU (ReLU²)** activation in its feed-forward layers, but llama.cpp's `bitnet` graph uses **SiLU**, and these GGUF files do not record which activation to use. A pull request on Microsoft's BitNet repository ([microsoft/BitNet #605](https://github.com/microsoft/BitNet/pull/605), fixing issue #602) describes the same bug and a ~6× perplexity regression on every backend. It is known upstream but **not fixed** as of November 2026: a one-line llama.cpp PR ([#25885](https://github.com/ggml-org/llama.cpp/pull/25885)) and a PR reading the activation from GGUF metadata ([#26751](https://github.com/ggml-org/llama.cpp/pull/26751)) were both closed unmerged, because the same `bitnet` architecture also serves older BitNet models that really use SiLU. Tracking issue: [microsoft/BitNet #588](https://github.com/microsoft/BitNet/issues/588).

**Fix (local patch, mainline llama.cpp):**
```bash
cd ~/llama.cpp
grep -n "LLM_FFN_" src/models/bitnet.cpp        # shows LLM_FFN_SILU
sed -i 's/LLM_FFN_SILU/LLM_FFN_RELU_SQR/' src/models/bitnet.cpp
cmake --build build -j 4
```
Perplexity dropped from **101.84 to 16.91** (6.0×). Note: this patch is right for the 2B4T files used here; older BitNet models that genuinely use SiLU would need the original line. ik_llama.cpp produced identical (looping) output, so it has the same issue; it was **not** patched here. The activation is a negligible share of the computation, so the speed and energy figures in this section (most measured before the fix) are expected to stand.

#### 11.2 Quality (WikiText-2, `llama-perplexity -c 512 --chunks 20`)

| Model | Perplexity (lower is better) |
|---|---|
| BitNet 2B TQ2_0, unpatched (SiLU) | 101.84 ± 5.14 |
| **BitNet 2B TQ2_0, patched (ReLU²)** | **16.91 ± 0.71** |
| **BitNet 2B TQ2_0 + Q8_0 output layer, patched** | **16.91 ± 0.71** |
| **Llama 3.2 1B Q4_0** | **16.95 ± 0.69** |
| Llama 3.2 3B Q4_0 | 11.93 ± 0.47 |

- On WikiText-2, BitNet 2B matches **Llama 3.2 1B**, not 3B. Section 11.5 therefore compares against the 1B.
- Perplexity is one measure. On Microsoft's own downstream benchmarks (reasoning, QA, etc.) BitNet b1.58 2B averages 54.2 against 44.9 for Llama 3.2 1B ([technical report](https://arxiv.org/abs/2504.12285)), so "equal quality" depends on the metric.

#### 11.3 Speed: output layer and runtime repacking (A720s + X4, 4 threads)

The model's output layer (128,256 × 2,560, shared with the input embeddings) is stored in **f16: 626 MiB**, more than all 30 ternary layers together (512 MiB). Since generation is memory-bound, that layer was quantised to Q8_0 (332.7 MiB) with the ternary layers unchanged:

```bash
./build/bin/llama-quantize --allow-requantize --token-embedding-type q8_0 bitnet-2b4t-tq2_0.gguf bitnet-tq2-emb8.gguf TQ2_0
```

ik_llama.cpp cannot read mainline TQ2_0 (it reports `unsupported type 35`). Its script `scripts/repack_ternary_to_prism.py` converts TQ2_0 to its PQ2_0 losslessly in seconds (standard-library Python only); greedy output matched mainline token for token. ik_llama labels the result "IQ4_KSS" (a mislabel from the file-type tag) and `llama-bench` reports 2.74 B params / 1.74 GiB because of how it handles the shared output layer at load time.

```bash
python3 ~/ik_llama.cpp/scripts/repack_ternary_to_prism.py bitnet-tq2-emb8.gguf bitnet-pq2-emb8.gguf
./build/bin/llama-bench -m bitnet-pq2-emb8.gguf -p 4,512 -n 128 -t 4 -r 3 -rtr 1   # other ik tools: -rtr with no number
```

| BitNet 2B setup | pp4 | pp512 | tg128 |
|---|---|---|---|
| Mainline TQ2_0, 3× A720 | 24.88 ± 0.40 | 33.26 ± 0.05 | 14.93 ± 0.55 |
| Mainline TQ2_0 (f16 output) | 37.87 ± 1.91 | 50.17 ± 0.83 | 18.20 ± 0.31 |
| Mainline TQ2_0 + Q8_0 output | 40.17 ± 0.48 | 48.37 ± 0.72 | 20.06 ± 1.55 |
| ik_llama PQ2_0, no repacking | 7.14 ± 0.08 | 14.06 ± 0.06 | 1.95 ± 0.59 |
| ik_llama PQ2_0, `-rtr 1` | 56.68 ± 1.49 | 91.88 ± 1.16 | 19.72 ± 0.27 |
| **ik_llama PQ2_0 + Q8_0 output, `-rtr 1`** | **66.91 ± 0.31** | **92.02 ± 0.35** | **25.25 ± 0.28** |
| *Llama 3.2 1B Q4_0, 3× A720 (section 4)* | – | *168.52* | *28.63* |

- On the A720s alone TQ2_0 is slightly compute-bound; adding the X4 brings mainline to the memory ceiling.
- **Runtime repacking works on ARM:** ik_llama's prompt processing went from 14 to 92 tok/s (1.8× mainline).
- **The smaller output layer only pays off with the faster kernel:** in mainline it gave +10% (the ternary kernel becomes the limit again); with ik_llama's repacking it gave **25.3 tok/s, 1.39× the original**, at ~23 GB/s effective, back at the memory ceiling. Its quality cost was zero (11.2).

#### 11.4 Batched generation (`llama-batched-bench -npp 128 -ntg 128`, total S_TG in tok/s)

| Parallel requests | BitNet, mainline (A720 + X4) | **BitNet, ik `-rtr` + Q8_0 output (A720 + X4)** | Llama 3.2 3B Q4_0 (A720 + X4) | Llama 3.2 1B Q4_0 (3× A720, section 7) |
|---|---|---|---|---|
| 1 | 19.76 | **26.42** | 14.05 | 27.95 |
| 2 | 25.18 | **40.74** | 17.43 | 34.65 |
| 4 | 29.06 | **50.51** | 41.59 | 78.08 |
| 8 | 31.39 | **51.35** | 43.89 | 80.10 |

A cold rerun of mainline BitNet reproduced it within 2–4% (20.15 / 25.84 / 30.44 / 32.22). Llama 3B Q4_0 jumps at 4 requests (17.4 → 41.6), the same pattern as the 1B model in section 7, supporting the explanation that Q4_0's repacked ARM kernels work in blocks of 4.

#### 11.5 Energy per token (method as in section 8; idle baseline 3.70 W)

**Against Llama 3.2 3B Q4_0** (a stronger model by perplexity):

| Test | Model (cores) | Runs | Generation | Phone power | J/token (inference) | J/token (whole phone) |
|---|---|---|---|---|---|---|
| 1 request (`tg512`) | BitNet mainline, f16 output (A720 + X4) | 3 | 16.6 tok/s | 10.25 W | 0.40 | 0.62 |
| 1 request (`tg512`) | Llama 3B Q4_0 (3× A720) | 2 | 11.5 tok/s | 12.28 W | 0.75 | 1.07 |
| 4 requests (`-ntg 256`) | BitNet mainline (A720 + X4) | 2 | 27.1 / 29.4 tok/s | 9.59 / 10.31 W | 0.22 | 0.35 |
| 4 requests (`-ntg 256`) | Llama 3B Q4_0 (A720 + X4) | 1 | 37.7 tok/s | 11.09 W | 0.20 | 0.29 |

Individual 1-request runs: BitNet 16.67 / 16.25 / 16.77 tok/s at 10.22 / 10.09 / 10.43 W; Llama 3B 11.44 / 11.59 tok/s at 11.72 / 12.83 W (the coldest start, 23.7 °C, drew the most power).

**At equal WikiText perplexity: BitNet best setup vs Llama 3.2 1B Q4_0** (1 request, `tg512`):

| Model (cores) | Run | Generation | Phone power | J/token (inference) | J/token (whole phone) | Temp (start → end) |
|---|---|---|---|---|---|---|
| Llama 3.2 1B Q4_0 (3× A720) | 1 | 27.3 tok/s | 13.17 W | 0.35 | 0.48 | not logged |
| | 2 | 29.56 ± 0.60 tok/s | 11.84 W | 0.28 | 0.40 | 22.6 → 24.4 °C |
| | **avg** | **28.4 tok/s** | | **0.31** | **0.44** | |
| BitNet ik `-rtr` + Q8_0 output (A720 + X4) | 1 | 22.74 ± 5.56 tok/s | 10.70 W | 0.31 | 0.47 | 24.3 → 29.8 °C |
| | 2 | 24.96 ± 0.92 tok/s | 11.95 W | 0.33 | 0.48 | 23.5 → 24.8 °C |
| | **avg** | **23.9 tok/s** | | **0.32** | **0.48** | |

**Findings:**
- Against Llama 3.2 **3B**, BitNet used ~1.7–1.9× less energy per token for one request, but the 3B is clearly better by perplexity (11.9 vs 16.9), so that is not a like-for-like comparison.
- **At equal WikiText perplexity, Llama 3.2 1B Q4_0 is as good or better on this phone:** about 1.2× faster generation, 1.8× faster prompts, 1.5× more throughput at 4 requests, and about the same energy per token (0.31 vs 0.32 J inference; 0.44 vs 0.48 J whole phone). The cleanest cold pair (run 2 each) favours the 1B (0.28 vs 0.33 J).
- **Why ternary does not win here:** at 2 B scale the non-ternary parts dominate the bytes read per token. The shared embedding/output layer (128,256 × 2,560) is 333 MiB even at Q8_0, against 512 MiB for all ternary layers, so BitNet's weights (0.83 GiB) end up about 16% *larger* than Llama 3.2 1B Q4_0's (0.71 GiB). BitNet also needs the X4 to reach the memory ceiling, and its batched kernels are slower than Q4_0's i8mm path. The ternary layers' share, and so their advantage, should grow with model size.
- **Throttling and efficiency:** a first Llama 3B 4-request run, started hot, fell to 25.27 tok/s (−33% against the clean 37.68) at only 7.15 W, ending at 37.4 °C: Android had cut the clock speeds. Its energy per token was about the same for the whole phone (0.28 vs 0.29 J) and lower after subtracting idle power (0.14 vs 0.20 J inference). Since the inference figure depends on the idle baseline, this hints at, but does not show, better efficiency at lower clocks.
- **Practical results that do stand:** the ReLU² fix (essential for anyone running these files in llama.cpp), a free Q8_0 output layer, and ik_llama's runtime repacking working on ARM (25.3 tok/s, 92 tok/s prompts).

Caveats: two runs per configuration in the equal-quality comparison; perplexity on WikiText-2 only; ik_llama was not patched for ReLU² (its outputs are degraded, its timings are not expected to change); most speed tests were run before the fix.

---

## Limitations

- One device, one build. Results will differ with other llama.cpp versions, Android versions and drivers.
- Power comes from the phone's battery gauge via Termux:API (~1.5 s sampling, ~±0.4 W run-to-run), not an external meter.
- Speculative decoding was tested on one code prompt and one prose prompt, 256 tokens each.
- Room temperature was not logged; phone (battery) temperature was logged only for later power runs, and earlier cooldowns were judged by touch.
- 3–5 repetitions per speed test; power tests had 1–3 runs per setup.
- Bandwidth figures are approximations derived from file size × generation speed.

## Open questions / ideas for contributions

- **Cross-check the power figures** with a USB-C power meter, and measure power for speculative decoding.
- **Speculative decoding refinements**: draft on the GPU, Llama 3.1 8B in Q4_0, more prompts, and longer outputs.
- **Batching at longer contexts** (e.g. 1,000+ token conversations) and with larger models (3B, 7B), and confirming why a batch of 2 underperforms.
- **Why batch sizes behave as they do** (9 vs 8, 12 vs 16) in llama.cpp's ARM CPU kernels.
- **Snapdragon comparison** (e.g. Snapdragon 8 Elite / 8 Elite Gen 5 with Adreno and llama.cpp's OpenCL backend) using the same commands.
- **Fan or passive cooling** vs none, to quantify sustained-performance gains.
- Which other Mali GPUs (beyond the G715 here and the G925 in llama.cpp #30006) show the cooperative-matrix slowdown.
- Gemma 3n and other mobile-optimised models.
- **Get the BitNet ReLU² fix merged upstream** (an activation key in the GGUF that keeps SiLU as the default for older BitNet models; see 11.1 for the closed attempts) and patch ik_llama.cpp the same way.
- **Microsoft's bitnet.cpp** (ARM-specific kernels) on the same phone, and **larger ternary models**, where the ternary layers dominate the bytes read per token.
- **Downstream task benchmarks** (not only WikiText perplexity) for BitNet 2B vs Llama 3.2 1B.
- **Lower CPU clocks for efficiency:** a throttled run used about the same energy per token for the whole phone and less after subtracting idle power (section 11.5). Capping frequencies (needs root) would test this directly.
- Output quality of the sub-4-bit formats alongside their speed.

If you run these tests on another phone, please open an issue or pull request with your results table, device, SoC and llama.cpp build number.

## Future roadmap

These benchmarks are the baseline for an open-source framework for running AI models efficiently on hardware people already own, starting with phones.

### In progress

- **Software:** lightweight wrappers built on what these benchmarks found: running the CPU and GPU together (section 8: ~111 tok/s at ~0.11 J per token) and choosing batch and draft sizes that match the kernels' fixed block sizes (section 9), for faster local generation and less battery drain.
- **Hardware:** ternary (−1, 0, +1) dot-product units, designed with AI help and formally proven correct, are moving from simulation onto an FPGA board. With ternary weights a multiply is just a choice between +x, 0 and −x, so these units need no multipliers.

*If you work on low-power mobile inference or ternary hardware, watch this repository for upcoming releases and documentation.*

## Licence

Data and text: CC BY 4.0. Commands and scripts: MIT.
