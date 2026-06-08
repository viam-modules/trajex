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

    ::viam::sdk::ProtoStruct get_status() override {
        return {};
    }

    static std::vector<std::string> validate(const ::viam::sdk::ResourceConfig& cfg);

   private:
    struct config {
#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
        std::vector<std::string> generator_sequence = {"totg", "legacy"};
#else
        std::vector<std::string> generator_sequence = {"totg"};
#endif
        bool segment_for_trajex = false;

        static config from_resource_config(const ::viam::sdk::ResourceConfig& cfg);
    };

    const config config_;
};

}  // namespace viam::trajex::service
