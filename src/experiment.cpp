#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>


#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/service/hlo_parser.h"
#include "absl/status/statusor.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"

int main(int argc, char* argv[]) {
    // parse command line arguments
    std::string hlo_file_path;
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <hlo_file_path>" << std::endl;
        return 1;
    } else {
        hlo_file_path = argv[1];
    }

    // load hlo file
    std::ifstream hlo_file(hlo_file_path);
    std::string hlo_raw_str((std::istreambuf_iterator<char>(hlo_file)),
                            std::istreambuf_iterator<char>());

    std::unique_ptr<xla::HloModule> module = xla::ParseAndReturnUnverifiedModule(hlo_raw_str).value();
    xla::XlaComputation xla_computation(module->ToProto());

    // create a PJRT client (CPU!)
    std::unique_ptr<xla::PjRtClient> client =
      xla::GetTfrtCpuClient(/*asynchronous=*/false).value();

    xla::CompileOptions compile_options;
    std::unique_ptr<xla::PjRtLoadedExecutable> executable =
      client->Compile(xla_computation, compile_options).value();


    // Execute the computation
    // Prepare inputs.

    std::cout << "creating literals" << std::endl;
    xla::Literal literal_x =
      xla::LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});
    xla::Literal literal_y =
        xla::LiteralUtil::CreateR2<float>({{1.0f, 1.0f}, {1.0f, 1.0f}});
    xla::Literal literal_z =
        xla::LiteralUtil::CreateR2<float>({{10.0f, 1.0f}, {1.0f, 1.0f}});
  
    std::cout << "creating buffers" << std::endl;
    std::unique_ptr<xla::PjRtBuffer> param_x =
        client->BufferFromHostLiteral(literal_x, client->addressable_devices()[0])
            .value();
    std::unique_ptr<xla::PjRtBuffer> param_y =
        client->BufferFromHostLiteral(literal_y, client->addressable_devices()[0])
            .value();
    std::unique_ptr<xla::PjRtBuffer> param_z =
        client->BufferFromHostLiteral(literal_z, client->addressable_devices()[0])
            .value();

    std::cout << "executing" << std::endl;
    // Execute on CPU.
    xla::ExecuteOptions execute_options;

    // One vector<buffer> for each device.
    std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results =
        executable->Execute({{param_x.get(), param_y.get(), param_z.get()}}, execute_options)
            .value();

    std::cout << "obtaining result" << std::endl;
    // Get result.
    std::shared_ptr<xla::Literal> result_literal =
        results[0][0]->ToLiteralSync().value();
      
    std::cout << "result: " << result_literal->ToString() << std::endl;

    return 0;
}