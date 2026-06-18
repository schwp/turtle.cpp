#include "quantize.hpp"

std::unordered_map<std::string, Tensor>
load_tensors(GGUFFile &gguf, const uint8_t *mapped_data) {
  std::unordered_map<std::string, Tensor> tensors;

  for (GGUFTensorInfo &info : gguf.tensors) {
    Tensor t;
    t.type = info.type;
    t.n_dimensions = info.n_dimensions;

    size_t n_elements = 1;
    for (uint32_t i = 0; i < info.n_dimensions; i++)
      n_elements *= info.dimensions[i];
    t.n_elements = n_elements;

    t.data = mapped_data + gguf.tensor_offset + info.offset;

    tensors[info.name] = t;
  }

  return tensors;
}
