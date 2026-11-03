// Convert the stateful Qwen3.6 language IR to explicit cache inputs/outputs.
// This is a helper for qwen36_stateless_bench.py; it does not modify the source model.

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/pass/stateful_to_stateless.hpp>

static void print_ports(const char *kind, const ov::OutputVector &ports) {
    std::cout << kind << " (" << ports.size() << ")\n";
    for (size_t i = 0; i < ports.size(); ++i) {
        const auto &port = ports[i];
        std::cout << "  " << i << "  " << port.get_any_name() << "  "
                  << port.get_element_type() << "  " << port.get_partial_shape() << "\n";
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: qwen36_stateless_export <stateful.xml> <stateless.xml>\n";
        return 2;
    }

    ov::Core core;
    auto model = core.read_model(argv[1]);
    std::cout << "before: variables=" << model->get_variables().size()
              << " sinks=" << model->get_sinks().size() << "\n";

    ov::pass::StatefulToStateless().run_on_model(model);
    model->validate_nodes_and_infer_types();
    std::cout << "after: variables=" << model->get_variables().size()
              << " sinks=" << model->get_sinks().size() << "\n";
    if (!model->get_variables().empty() || !model->get_sinks().empty()) {
        std::cerr << "stateful cache nodes remain after conversion\n";
        return 1;
    }

    // The exported model normally returns logits for every prompt token. A long-context
    // prefill would make this [batch, context, 248320] tensor enormous. Generation only needs
    // the final prompt position, so slice the hidden states before the LM head projection.
    if (model->get_results().empty()) {
        std::cerr << "unexpected Qwen3.6 logits result\n";
        return 1;
    }
    auto lm_head = model->get_results()[0]->get_input_node_shared_ptr(0);
    auto indices = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, std::vector<int64_t>{1});
    auto last_hidden = std::make_shared<ov::op::v8::Gather>(lm_head->input_value(0), indices, axis, 0);
    last_hidden->set_friendly_name("stateless_last_prompt_hidden");
    lm_head->input(0).replace_source_output(last_hidden->output(0));
    auto one = ov::op::v0::Constant::create(ov::element::i64, {}, std::vector<int64_t>{1});
    auto squeeze_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{1});
    auto squeezed_logits = std::make_shared<ov::op::v0::Squeeze>(lm_head->output(0), squeeze_axis);
    auto topk = std::make_shared<ov::op::v11::TopK>(
        squeezed_logits->output(0), one, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::NONE, ov::element::i64, false);
    topk->set_friendly_name("stateless_greedy_token");
    model->get_results()[0]->input(0).replace_source_output(topk->output(1));
    topk->output(1).get_tensor().set_names({"next_token_ids"});
    model->validate_nodes_and_infer_types();
    std::cout << "LM head input is sliced to the final sequence position; output is greedy token ID\n";

    print_ports("inputs", model->inputs());
    print_ports("outputs", model->outputs());
    ov::serialize(model, argv[2]);
    return 0;
}
