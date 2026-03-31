#include <vart/vart_memory.hpp>
#include <vart/vart_memory_impl_vvas.hpp>
#include <vart/vart_memory_types.hpp>

#include <vart/vart_videoframe.hpp>
#include <vart/vart_videoframe_impl_xrt.hpp>
#include <vart/vart_videoframe_types.hpp>

#include <vart/vart_device.hpp>
#include <vart/vart_inferresult_types.hpp>
#include <vart/vart_metaconvert.hpp>
#include <vart/vart_overlay.hpp>
#include <vart/vart_overlay_types.hpp>
#include <vart/vart_runner_factory.hpp>
#include <vart/vart_npu_tensor.hpp>

#include <stdlib.h>
using namespace std;

#define DEFAULT_DEVICE_INDEX 1
#define MEM_BANK_IDX 0

using namespace vart;
/**
 * @brief  Get the string representation of the vart memory layout.
 * @param layout The memory layout of the tensor.
 * @return string
 */
static string get_memory_layout_string(const MemoryLayout& layout) {
  switch (layout) {
    case MemoryLayout::NHW:
      return "NHW";
    case MemoryLayout::NHWC:
      return "NHWC";
    case MemoryLayout::NCHW:
      return "NCHW";
    case MemoryLayout::NHWC4:
      return "NHWC4";
    case MemoryLayout::NC4HW4:
      return "NC4HW4";
    case MemoryLayout::NC8HW8:
      return "NC8HW8";
    case MemoryLayout::HCWNC4:
      return "HCWNC4";
    case MemoryLayout::HCWNC8:
      return "HCWNC8";
    default:
      return "UNKNOWN";
  }
}

/**
 * @brief  Get the string representation of the vart data type.
 * @param data_type The data type of the tensor.
 * @return string
 */
static string get_data_type_string(const vart::DataType& data_type) {
  switch (data_type) {
    case vart::DataType::INT8:
      return "INT8";
    case vart::DataType::UINT8:
      return "UINT8";
    case vart::DataType::INT16:
      return "INT16";
    case vart::DataType::UINT16:
      return "UINT16";
    case vart::DataType::BF16:
      return "BF16";
    case vart::DataType::FP16:
      return "FP16";
    case vart::DataType::FLOAT32:
      return "FLOAT32";
    default:
      return "UNKNOWN";
  }
}

/**
 * @brief Get the string representation of the tensor shape.
 * @param shape The shape of the tensor.
 * @return string
 */
string get_shape_string(const vector<unsigned int>& shape) {
  string shape_str = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_str += to_string(shape[i]);
    if (i < shape.size() - 1) {
      shape_str += ", ";
    }
  }
  shape_str += ")";
  return shape_str;
}

// Read raw data
bool load_raw_int8_from_bin(const std::string& filename, void* data) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(data), size);
    return true;
}

bool save_raw_float(const std::string& filename, const std::vector<float>& data) {
    std::ofstream f(filename, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return true;
}
bool save_output_to_bin(const std::shared_ptr<xrt::bo>& out_bo,
                        const std::string& filename,
                        size_t tensor_size_bytes) {

    void* output_data = out_bo->map();

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot open output file " << filename << std::endl;
        return false;
    }

    ofs.write(reinterpret_cast<char*>(output_data), tensor_size_bytes);
    ofs.close();
    std::cout << "Output saved to " << filename << " (" <<std::dec<< tensor_size_bytes << " bytes)" << std::endl;
    return true;
}
int main(){
  /* vart Npu runner context */
  shared_ptr<vart::Runner> runner;
  string model_path="yolov8m_VINT8_skipNodes";
  string input_file="test_image_int8.bin";

  std::shared_ptr<xrt::device> device = std::make_shared<xrt::device>(DEFAULT_DEVICE_INDEX);

  std::vector<std::vector<vart::NpuTensor>> npu_input_tensors;  //[batch][tensors]
  std::vector<std::vector<vart::NpuTensor>> npu_output_tensors;  //[batch][tensors]
	/* Create a VART Runner instance for inference */

  /* runner options */
  std::unordered_map<std::string, std::any> runner_options = {};
  runner_options["config_json"] = std::string("vitisai_config.json");
  /* Always operate in HW (zero copy mode) */
  runner_options["input_tensor_type"] = std::string("HW");
  runner_options["output_tensor_type"] = std::string("HW");
  runner_options["log_level"]=std::string("WARNING");
  runner_options["ai_analyzer_profiling"] = true;
  runner_options["ai_analyzer_visualization"] = true;

  runner = vart::RunnerFactory::create_runner(
      vart::RunnerType::VAIML, model_path, runner_options);

  auto input_tensors_info = runner->get_tensors_info(
      vart::TensorDirection::INPUT, vart::TensorType::HW);
  auto output_tensors_info = runner->get_tensors_info(
      vart::TensorDirection::OUTPUT, vart::TensorType::HW);
  std::cout<<"input tensor batch_size:"<<runner->get_batch_size()<<std::endl;
  std::cout<<"input tensor number:"<<runner->get_num_input_tensors()<<std::endl;
  std::cout<<"output tensor number:"<<runner->get_num_output_tensors()<<std::endl;
  std::cout<<"Input tensor name: "<<input_tensors_info[0].name<<std::endl;
  std::cout<<"Input tensor shape: "<<get_shape_string(input_tensors_info[0].shape)<<std::endl;
  std::cout<<"Input tensor size in bytes: "<<input_tensors_info[0].size_in_bytes<<std::endl;
  std::cout<<"Input quantization scale: "<<runner->get_quant_parameters(input_tensors_info[0].name).scale<<std::endl;
  std::cout<<"Input quantization zero_point: "<<runner->get_quant_parameters(input_tensors_info[0].name).zero_point<<std::endl;
  std::cout<<"Output tensor name: "<<output_tensors_info[0].name<<std::endl;
  std::cout<<"Output tensor shape: "<<get_shape_string(output_tensors_info[0].shape)<<std::endl;
  std::cout<<"Output tensor size in bytes: "<<output_tensors_info[0].size_in_bytes<<std::endl;
  std::cout<<"Output quantization_factor: "<<runner->get_quant_parameters(output_tensors_info[0].name).scale<<std::endl;
  std::cout<<"Output quantization zero_point: "<<runner->get_quant_parameters(output_tensors_info[0].name).zero_point<<std::endl;
  std::cout<<"Memory layout:"<<get_memory_layout_string(input_tensors_info[0].memory_layout)<<std::endl;
  std::cout<<"Memory data type:"<<get_data_type_string(input_tensors_info[0].data_type)<<std::endl;

  /* We support only 1 input tensor */
  auto tensor_mem_type = vart::MemoryType::XRT_BO;
  auto tensor_type = vart::TensorType::HW;
  std::shared_ptr<xrt::bo> in_bo = std::make_shared<xrt::bo>(*device, input_tensors_info[0].size_in_bytes, MEM_BANK_IDX);
  void* input_buffer = in_bo->map();
  load_raw_int8_from_bin(input_file, input_buffer);
  in_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  vart::NpuTensor in_tensor(input_tensors_info[0], in_bo.get(), tensor_mem_type);
  std::vector<vart::NpuTensor> input_tensors;
  input_tensors.push_back(std::move(in_tensor));
  npu_input_tensors.push_back(std::move(input_tensors));

  std::shared_ptr<xrt::bo> out_bo(new xrt::bo(*device, output_tensors_info[0].size_in_bytes, MEM_BANK_IDX));
  void *output_buffer=out_bo->map();
  vart::NpuTensor out_tensor(output_tensors_info[0], out_bo.get(), tensor_mem_type);
  std::vector<vart::NpuTensor> output_tensors;
  output_tensors.push_back(std::move(out_tensor));
  npu_output_tensors.push_back(std::move(output_tensors));
  std::cout<<"debug point 2"<<std::endl;

  std::cout << "Input tensor size: " << input_tensors_info[0].size_in_bytes << std::endl;
  std::cout << "Output tensor size: " << output_tensors_info[0].size_in_bytes << std::endl;
  std::cout << "Input buffer addr: " << std::hex << in_bo->address() << std::endl;
  std::cout << "Output buffer addr: " << std::hex << out_bo->address() << std::endl;
  std::cout << "MEM_BANK_IDX: " << MEM_BANK_IDX << std::endl;
  std::cout << "TensorType: " << (int)tensor_type << "  MemType: " << (int)tensor_mem_type << std::endl;
  sleep(2);
  try {
    auto start = chrono::high_resolution_clock::now();
    int REPEAT=1000;
    for(int i=0;i<REPEAT;i++){
        auto ret = runner->execute(npu_input_tensors, npu_output_tensors);
        //if (vart::StatusCode::SUCCESS != ret) {
        //  std::cout<<"Inference execution failed"<<std::endl;;
        //  return false;
        //}
    }
    auto end = chrono::high_resolution_clock::now();
    auto total_infer_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double infer_ms = total_infer_time / 1000.0 / REPEAT;
    double fps = 1000000.0 / total_infer_time * REPEAT;
    std::cout << "Inference time: " << infer_ms << " ms" << std::endl;
    std::cout << "FPS: " << fps << std::endl;

  } catch (const exception& e) {
    std::cout<<"Inference execution failed: "<<e.what()<<std::endl;
    return false;
  }
  // Device -> Host
  out_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  save_output_to_bin(out_bo, "output0_vart.bin", output_tensors_info[0].size_in_bytes);
  std::cout<<"Inference Done!"<<std::endl;
}
