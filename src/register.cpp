#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <torch/csrc/jit/custom_operator.h>

using namespace torch.jit;
PYBIND_MODULE(pointwise_compiler, )
const auto pointwise_compiler_symbol = Symbol::fromQualString("pw::CompilationGroup");

RegisterPass pass([pointwise_compiler_symbol](std::shared_ptr<Graph>& g)) {
    CustomFuseGraph(g, PointwiseCompiler::supported, pointwise_compiler_symbol);
}

void CustomFuseGraph(std::shared_ptr<Graph>& g, bool(*)(Node*) callback, Symbol s)

auto options = c10::OperatorOptions();
options.setAliasAnalysis(AliasAnalysisKind::PURE);

RegisterOperators op({Operator(
    pointwise_compiler_symbol,
    [](const Node* node) {
        auto compiler = std::make_shared<PointwiseCompiler>(node);
        return [compiler](Stack& stack) {
            compiler->run(stack);
            return 0;
        };
    },
    options)});
