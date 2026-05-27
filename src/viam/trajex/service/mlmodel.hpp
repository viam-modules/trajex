#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <viam/sdk/config/resource.hpp>
#include <viam/sdk/services/mlmodel.hpp>

namespace viam::trajex::service {

class mlmodel final : public ::viam::sdk::MLModelService {
   public:
    mlmodel(::viam::sdk::Dependencies deps, ::viam::sdk::ResourceConfig config);

    std::shared_ptr<named_tensor_views> infer(const named_tensor_views& inputs, const ::viam::sdk::ProtoStruct& extra) override;

    struct metadata metadata(const ::viam::sdk::ProtoStruct& extra) override;

    // SDK lifecycle hook (added in viam-cpp-sdk after v0.31.0). We don't
    // surface any service-level status today; an empty ProtoStruct
    // matches the SDK's bundled examples.
    ::viam::sdk::ProtoStruct get_status() override {
        return {};
    }

    static std::vector<std::string> validate(const ::viam::sdk::ResourceConfig& cfg);

   private:
    struct config {
        std::vector<std::string> generator_sequence = {"totg", "legacy"};
        bool segment_for_trajex = false;
    };

    static config parse_config(const ::viam::sdk::ResourceConfig& cfg);

    const config config_;
};

}  // namespace viam::trajex::service
