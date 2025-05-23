#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/argument_spec.h>

#include <asmjit/asmjit.h>

using CompiledCode = std::function<std::vector<torch::jit::IValue>(
    at::ArrayRef<torch::jit::IValue>&)>;

class RegisterManager;

class PointwiseCompiler {
    public:
     PointwiseCompiler(const torch::jit::Node* node) : subgraph_(node->g(torch::jit::attr::Subgraph)) {}
     void run(torch::jit::Stack& stack);

     static bool supported(const torch::jit::Node* node);
    
    private:
     void emitOperation(
        const torch::jit::Node* node,
        const std::std<const torch::jit::Node*>& seen,
        asmjit::X86Assembler& assembler,
        RegisterManager& reg_manager);
     CompiledCode compile(at::ArrayRef<torch::jit::IValue>&);
     std::shared_ptr<torch::jit::Graph> subgraph_;
     std::unordered_map<torch::jit::CompleteArgumentSpec, CompiledCode> cache_;
     asmjit::JitRuntime jit_runtime_;
};