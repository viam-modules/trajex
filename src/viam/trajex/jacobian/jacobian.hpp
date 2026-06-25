#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

namespace viam::trajex::jacobian {

// TODO(RSDK-14104): Remove duplicated code that exists in the viam-cpp-sdk

///
/// A validated URDF-style serial kinematic chain, parsed from an (n, 10)
/// model-table tensor and held in chain order.
///
class kinematic_chain {
   public:
    ///
    /// Parses an (n, 10) tensor in the viam::sdk::ModelTable format into a
    /// kinematic chain.
    ///
    /// @param tensor (n, 10) tensor in the viam::sdk::ModelTable format
    /// @return Validated kinematic chain
    /// @throws std::invalid_argument on non-2D input, wrong column count, a
    ///         non-integer joint-type encoding, an empty table, an unsupported
    ///         joint type (continuous or prismatic), or a revolute row with
    ///         zero-magnitude axis
    ///
    [[nodiscard]] static kinematic_chain from(const xt::xarray<double>& tensor);

    ///
    /// Computes the geometric Jacobian at joint positions q.
    ///
    /// @param q (N_actuated,) vector with one element per revolute row in the
    ///        table, in chain order. Fixed rows do not consume a q entry.
    /// @return A (6, N_actuated) xarray where rows 0..2 are the
    ///         linear-velocity columns J_v_i = w_i x (p_e - p_i) and rows 3..5
    ///         are the angular-velocity columns J_w_i = w_i, with w_i the
    ///         world-frame axis of revolute joint i, p_i its world position,
    ///         and p_e the end-effector position.
    /// @throws std::invalid_argument on q-size mismatch
    ///
    [[nodiscard]] xt::xarray<double> jacobian(const xt::xarray<double>& q) const;

    ///
    /// Computes the linear-velocity block of the geometric Jacobian at joint
    /// positions q.
    ///
    /// @param q (N_actuated,) vector with one element per revolute row in the
    ///        table, in chain order. Fixed rows do not consume a q entry.
    /// @return A (3, N_actuated) xarray of linear-velocity columns.
    /// @throws std::invalid_argument on q-size mismatch
    ///
    [[nodiscard]] xt::xarray<double> linear_jacobian(const xt::xarray<double>& q) const;

   private:
    // URDF joint type, restricted to arm-relevant joints. Underlying values
    // are the column-9 wire encoding accepted by `from`, and match
    // viam::sdk::ModelTable::JointType.
    enum class joint_type_ : std::uint8_t {
        k_revolute = 0,
        k_continuous = 1,
        k_prismatic = 2,
        k_fixed = 3,
    };

    // One row of the model table: the per-joint URDF fields. xyz/rpy are the
    // joint origin relative to the parent link (rpy is fixed-axis XYZ); axis
    // is the joint axis in the local frame.
    struct joint_row {
        std::array<double, 3> xyz{};
        std::array<double, 3> rpy{};
        std::array<double, 3> axis{};
        joint_type_ type = joint_type_::k_fixed;
    };

    // Per-revolute-joint world-frame axes and origins plus the end-effector
    // position.
    struct chain_state;

    // Validates the rows (joint types, axes) and counts actuated joints; all
    // public construction funnels through here via `from`.
    explicit kinematic_chain(std::vector<joint_row> rows);

    // Evaluates the forward kinematics at joint positions q, capturing the
    // per-joint quantities the Jacobian assemblies need. Throws
    // std::invalid_argument on q-size mismatch.
    chain_state compute_chain_state_(const xt::xarray<double>& q) const;

    std::vector<joint_row> rows_;
    std::size_t actuated_count_ = 0;
};

}  // namespace viam::trajex::jacobian
