#pragma once

#include <array>
#include <cstddef>
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
/// Build one with the `from` factory. Construction validates the chain, so a
/// constructed chain always satisfies the class invariant: every row is
/// revolute or fixed, and every revolute row has a non-zero axis. Parse once
/// and reuse across `jacobian()` calls to avoid re-validating the tensor on
/// every evaluation.
///
/// **Thread safety**: All const methods are thread-safe for concurrent access.
///
/// Example usage:
/// @code
///   const auto chain = kinematic_chain::from(tensor);  // (n, 10) tensor
///   xt::xarray<double> J = chain.jacobian(q);          // (6, N_actuated)
/// @endcode
///
class kinematic_chain {
   public:
    ///
    /// Parses an (n, 10) tensor in the viam::sdk::ModelTable format into a
    /// kinematic chain.
    ///
    /// Columns: 0..2 xyz, 3..5 rpy, 6..8 axis, 9 joint type encoded as
    /// revolute=0, continuous=1, prismatic=2, fixed=3 (matching
    /// viam::sdk::ModelTable::JointType).
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
    /// Equivalent to rows 0..2 of `jacobian(q)`: column i is
    /// J_v_i = w_i x (p_e - p_i). Use when only the Cartesian linear velocity
    /// is needed.
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
    enum class joint_type {
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
        joint_type type = joint_type::k_fixed;
    };

    // Accumulated chain-walk results shared by both Jacobian assemblies;
    // defined in jacobian.cpp.
    struct chain_state;

    // Validates the rows (joint types, axes) and counts actuated joints; all
    // public construction funnels through here via `from`.
    explicit kinematic_chain(std::vector<joint_row> rows);

    // Walks the chain at joint positions q. Throws std::invalid_argument on
    // q-size mismatch.
    chain_state walk_chain(const xt::xarray<double>& q) const;

    std::vector<joint_row> rows_;
    std::size_t actuated_count_ = 0;
};

}  // namespace viam::trajex::jacobian
