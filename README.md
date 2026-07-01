# turtle.cpp

A minimal LLM inference engine written from scratch in C++. Loads GGUF models, 
dequantizes Q4_0/Q6_K weights on-the-fly, and generates text through a full
transformer forward pass. No dependencies on llama.cpp, PyTorch, or any ML framework
have been used to create the engine.

Built to understand every layer and methods of the inference stack: from parsing
bytes off disk to producing English one token at a time.

> [!NOTE]
> The engine can still be improved by adding more quantization formats, hardware
> acceleration, and model architectures. Currently, it supports only Q4_0/Q8_0/Q6_K/F16/F32
> quantization, is based for CPU inference (with OpenMP/AVX2 optimizations, no
> GPU acceleration), and is focused on LlaMA-architecture models.

## Demo

```
$ ./turtle_cpp "../data/tinyllama-1.1b-chat-v1.0.Q4_0.gguf" --max-tokens 50
Threads: 4
Sampling: greedy
Prompt: "The capital of France is"
 a city in the region of France. It is located in the north of Paris, France, and is a city of Paris, and is the capital of Paris, France.<0x0A>190101, and is a French, France,
Prefill: 6 tokens in 779.0 ms (7.7 tok/s)
Generation: 50 tokens in 6585.6 ms (7.59 tok/s)
```

## Performance

Benchmarked on **AMD Ryzen 5 2600 Six-Core Processor**, [TinyLlama-1.1B Q4_0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf),
50-token generation.

| Optimization            | tok/s | Speedup |
|-------------------------|-------|---------|
| Baseline (-O0)          | 0.29  | 1.0×    |
| Compiler (-O2)          | 1.05  | 3.6×    |
| + OpenMP (8 threads)    | 6.93  | 23.9×   |
| + AVX2 SIMD             | 7.27  | 25.1×   |
| llama.cpp (reference)   | ~20   | ~69×    |

## What's implemented

Everything from raw bytes on disk to generated text — no external ML libraries:

**GGUF parser** — reads the binary file format: header, metadata key-value pairs, tensor info array, and alignment. Supports version 3. Model configuration (hidden size, number of heads, RoPE theta, etc.) is extracted from metadata keys, making the same code work for any LLaMA-family model.

**Quantized tensor access** — model weights stay in their compressed format in memory (via mmap). No full-model dequantization at load time. Q4_0 weights occupy 0.56 bytes per parameter (7× compression vs FP32), Q6_K uses 0.82 bytes (4.9× compression). Tensor data is accessed through zero-copy pointers into the memory-mapped file.

**Fused dequantization** — dot products between FP32 activations and quantized weights are computed in a single pass. Each quantized block is dequantized into CPU registers, multiplied with the input, and accumulated — the dequantized values never touch main memory. This is what makes quantized inference memory-efficient.

**Transformer forward pass:**
- RMSNorm with double-precision accumulation for numerical stability
- Rotary Position Embeddings (RoPE) with configurable theta (10k for TinyLlama, 500k for Llama 3.2)
- Multi-head attention with Grouped Query Attention (GQA): 32 query heads share 4 KV heads
- KV cache: pre-allocated flat buffer, O(1) per-token append, supports sequences up to context length
- SwiGLU MLP: gated activation with SiLU nonlinearity
- Residual connections after attention and MLP blocks

**Text generation** — autoregressive loop with prefill (process prompt, populate KV cache) and generation (produce one token per step) phases. Supports greedy decoding (argmax) and temperature + top-p sampling.


## Project structure

```
turtle.cpp/
├── include/
│   ├── gguf.hpp          # GGUF file format types + parser declaration
│   ├── quantize.hpp      # block structs, dequant, vec_dot, matmul
│   ├── ops.hpp           # rmsnorm, rope, softmax, silu, add
│   ├── model.hpp         # Model/KVCache structs, forward pass
│   └── cli.h             # argument parsing
├── src/
│   ├── gguf.cpp          # GGUF parser: header, metadata, tensor info, mmap
│   ├── quantize.cpp      # Q4_0/Q6_K/Q8_0 dequantization + fused dot products
│   ├── ops.cpp           # transformer building blocks
│   ├── model.cpp         # model loading, attention, MLP, forward, generate
│   ├── cli.cpp           # CLI argument parsing
│   └── main.cpp          # entry point
├── tests/
│   └── test_ops.cpp      # GTest: all ops verified against PyTorch (libtorch)
├── CMakeLists.txt
└── README.md
```

---

## Build

```bash
git clone https://github.com/schwp/turtle.cpp.git
cd turtle.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requirements: CMake $\geq$ 3.18, C++17 compiler with OpenMP and AVX2 support.

## Usage

Download a GGUF model and run the inference engine. Here are some examples if you
want to try it out :
```bash
./turtle_cpp tinyllama.gguf --max-tokens 50

./turtle_cpp tinyllama.gguf --max-tokens 100 --sample --temperature 0.8 --top-p 0.9

./turtle_cpp tinyllama.gguf --threads 8 --max-tokens 50
```

## Key lessons

**mmap is the entire memory strategy.** The model file is memory-mapped and tensor
pointers index directly into it. No malloc, no memcpy, no conversion pass. The OS
pages in data on demand. This is how a 637 MB model loads in milliseconds, you're
not reading the file, you're just creating a virtual address mapping.

**Dequantize inside the dot product, not before it.** A full FP32 copy of TinyLlama
would be 4.4 GB, larger than the GPU memory on many machines. By dequantizing 
block-by-block during the matmul inner loop, the FP32 values exist only in CPU
registers for one clock cycle. Memory usage equals the compressed file size, not 
the uncompressed model size.

**The Q6_K struct layout does not match the logical element ordering.** The 256
elements are split into two groups of 128, with low and high nibbles of `ql` mapping
to four interleaved sub-groups. Getting this layout wrong produces valid-looking
but numerically incorrect logits, the model generates real words but not coherent
text. This was the last bug fixed in this project and the hardest to diagnose.

**OpenMP gives 7× for one pragma line.** The matmul output loop is now in parallel,
each output element is an independent dot product. Adding `#pragma omp parallel for`
and `-fopenmp` gave more speedup than AVX2 SIMD.

**The Rule of Five matters for real.** The GGUF file handle owns a mmap pointer.
Without a move constructor and deleted copy constructor, returning a `GGUFFile`
from a function silently copies the pointer and the temporary's destructor frees
the mapping, every tensor pointer becomes dangling. This produced a segfault that
took longer to find than the entire forward pass implementation.

## Testing

```bash
cd build
ctest --output-on-failure
```

All ops (RMSNorm, RoPE, softmax, SiLU) are tested against PyTorch via libtorch
(automatically downloaded using CMake) to verify numerical correctness. RoPE is
specifically tested for the rotated-half layout used by LLaMA 2/3, position-0
identity, and magnitude preservation.

## Comparison with llama.cpp

This project is a learning exercise, not a production tool. Here are the main 
differences:

| Feature                | turtle.cpp          | llama.cpp               |
|------------------------|---------------------|-------------------------|
| Performance            | ~7 tok/s            | ~20+ tok/s              |
| GPU support            | No                  | CUDA, Metal, Vulkan     |
| Batch processing       | No                  | Yes                     |
| Memory allocator       | std::vector         | Custom pool allocator   |
| SIMD                   | Basic AVX2          | AVX-512, VNNI, ARM NEON |
| Quantization formats   | Q4_0, Q8_0, Q6_K    | 20+ formats             |
| Tokenizer              | Decode only         | Full BPE encode/decode  |
| Model support          | LLaMA family only   | LLaMA, Mistral, Phi, Gemma, ... |

The value of this project is not competing with llama.cpp, it's to understand what
llama.cpp does under the hood and why each optimization matters, and how it could be
helpful for some companies.

## References

- [GGUF format specification](https://huggingface.co/docs/hub/gguf)
- [gguf-tools project](https://github.com/antirez/gguf-tools)
- [ggml project](https://github.com/ggml-org/ggml)
- [LLaMA 2 paper](https://arxiv.org/abs/2307.09288) - Touvron et al.
- [RoPE - Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al.
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [llama2.c](https://github.com/karpathy/llama2.c) - Andrej Karpathy

## Author
- Pierre SCHWEITZER (schwp)
