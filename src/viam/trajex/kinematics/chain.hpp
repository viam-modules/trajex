#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::kinematics {

///
/// A validated serial kinematic chain, held in chain order.
///
/// Chains are assembled with chain::builder. A finished chain is immutable and copyable.
/// Each rotating or translating joint consumes one element of the joint vector q, in chain order;
/// fixed joints consume none. A rotating joint's element is an angle in radians, and a translating
/// joint's is a displacement in the length unit of the chain's translations.
///
class chain {
   public:
    ///
    /// The rigid transform from a joint's parent frame to the joint's own frame.
    ///
    struct joint_origin;

    ///
    /// The axis a revolute or continuous joint rotates about.
    ///
    struct rotation_axis;

    ///
    /// The axis a prismatic joint translates along.
    ///
    struct translation_axis;

    ///
    /// Assembles a kinematic chain one joint at a time.
    ///
    class builder;

    ///
    /// The position of a frame's origin, and the linear velocity and acceleration of that origin
    /// for a given joint velocity and acceleration. All three are expressed in the chain's base
    /// frame, in the length unit of the chain's translations.
    ///
    /// The acceleration is that of the origin as a moving point, J_v_dot * q_dot + J_v * q_ddot.
    /// The frame's orientation and angular motion are not reported. If they are added later, they
    /// will be named orientation, angular_velocity and angular_acceleration.
    ///
    struct frame_kinematics {
        std::array<double, 3> position{};             ///< Position of the frame's origin
        std::array<double, 3> linear_velocity{};      ///< Velocity of the frame's origin
        std::array<double, 3> linear_acceleration{};  ///< Acceleration of the frame's origin
    };

    ///
    /// Gets the number of elements in the joint vector q: one per rotating or translating joint.
    ///
    /// @return The chain's degrees of freedom
    ///
    [[nodiscard]] std::size_t dof() const noexcept {
        return actuated_count_;
    }

    ///
    /// Gets the number of joints in the chain, fixed ones included. There is one for each call
    /// made on the builder, and each has a frame that forward_kinematics can report on.
    ///
    /// @return The chain's joint count
    ///
    // Defined below the joint types, because joints_ cannot be used while they are incomplete.
    [[nodiscard]] std::size_t size() const noexcept;

    ///
    /// Reports each joint, from the base to the end effector, exactly as it was given to the
    /// builder.
    ///
    /// The visitor is called once per joint with the arguments of the builder overload that added
    /// it: (const joint_origin&) for a fixed joint, (const joint_origin&, const rotation_axis&) for
    /// a rotating one, and (const joint_origin&, const translation_axis&) for a translating one.
    /// Making the same calls on a builder rebuilds an identical chain, which is how a chain is
    /// serialized and restored.
    ///
    /// @param visitor Callable accepting all three argument lists
    ///
    template <typename Visitor>
    void visit_joints(Visitor&& visitor) const;

    ///
    /// Computes the geometric Jacobian at joint positions q.
    ///
    /// @param q (N_actuated,) vector with one element per rotating or
    ///        translating joint, in chain order. Fixed joints do not consume a
    ///        q entry.
    /// @return A (6, N_actuated) matrix where rows 0..2 are the
    ///         linear-velocity columns and rows 3..5 the angular-velocity
    ///         columns. With w_i the world-frame axis of joint i, p_i its world
    ///         position, and p_e the end-effector position, a rotating joint's
    ///         column is J_v_i = w_i x (p_e - p_i), J_w_i = w_i, and a
    ///         translating joint's is J_v_i = w_i, J_w_i = 0.
    /// @throws std::invalid_argument on q-size mismatch, or if q is not 1-dimensional
    ///
    /// A caller holding an xvector binds here with no copy.
    ///
    [[nodiscard]] xmatrix<> jacobian(const xvector<>& q) const;

    ///
    /// Computes the geometric Jacobian at joint positions q held in something other than an
    /// xvector.
    ///
    /// @throws std::invalid_argument if q is not 1-dimensional, plus everything the overload
    ///         above throws
    ///
    template <xrank_same_as_or_dynamic<xvector<>> Q>
    [[nodiscard]] xmatrix<> jacobian(const Q& q) const {
        require_rank_one_(q);
        return jacobian(xvector<>(q));
    }

    // Otherwise this converts into the xvector overload, which is the UB case.
    template <xexpression_like Q>
        requires(!xrank_same_as_or_dynamic<Q, xvector<>>)
    xmatrix<> jacobian(const Q&) const = delete;

    ///
    /// Computes the linear-velocity block of the geometric Jacobian at joint
    /// positions q.
    ///
    /// @param q (N_actuated,) vector with one element per rotating or
    ///        translating joint, in chain order. Fixed joints do not consume a
    ///        q entry.
    /// @return A (3, N_actuated) matrix of linear-velocity columns.
    /// @throws std::invalid_argument on q-size mismatch
    ///
    [[nodiscard]] xmatrix<> linear_jacobian(const xvector<>& q) const;

    ///
    /// As above, for a q the caller holds in something other than an xvector.
    ///
    /// @throws std::invalid_argument if q is not 1-dimensional, plus everything the overload
    ///         above throws
    ///
    template <xrank_same_as_or_dynamic<xvector<>> Q>
    [[nodiscard]] xmatrix<> linear_jacobian(const Q& q) const {
        require_rank_one_(q);
        return linear_jacobian(xvector<>(q));
    }

    template <xexpression_like Q>
        requires(!xrank_same_as_or_dynamic<Q, xvector<>>)
    xmatrix<> linear_jacobian(const Q&) const = delete;

    ///
    /// Computes the kinematics of one joint's frame, after that joint's motion, at joint positions
    /// q with joint velocity q_dot and joint acceleration q_ddot.
    ///
    /// The frame depends only on the joints up to and including the one named, so only the
    /// leading entries of q, q_dot and q_ddot that belong to those joints affect it. All three
    /// must still have one entry for every moving joint in the chain.
    ///
    /// Nothing here depends on what the derivatives are taken with respect to. If q_dot and
    /// q_ddot are derivatives along a path rather than in time, the velocity and acceleration
    /// returned are derivatives along the same path.
    ///
    /// @param joint Index of the joint, counting fixed joints, which must be less than size()
    /// @param q (N_actuated,) joint positions, in chain order
    /// @param q_dot (N_actuated,) joint velocities
    /// @param q_ddot (N_actuated,) joint accelerations
    /// @return The frame's position, linear velocity and linear acceleration
    /// @throws std::out_of_range if joint is not less than size()
    /// @throws std::invalid_argument on a size mismatch
    ///
    [[nodiscard]] frame_kinematics forward_kinematics(std::size_t joint,
                                                      const xvector<>& q,
                                                      const xvector<>& q_dot,
                                                      const xvector<>& q_ddot) const;

    ///
    /// As above, for arguments the caller holds in something other than an xvector.
    ///
    /// All three deduce independently, so a caller may mix. Three xvectors bind to the overload
    /// above instead.
    ///
    /// @throws std::invalid_argument if any argument is not 1-dimensional, plus everything the
    ///         overload above throws
    ///
    template <xrank_same_as_or_dynamic<xvector<>> Q, xrank_same_as_or_dynamic<xvector<>> QD, xrank_same_as_or_dynamic<xvector<>> QDD>
    [[nodiscard]] frame_kinematics forward_kinematics(std::size_t joint, const Q& q, const QD& q_dot, const QDD& q_ddot) const {
        require_rank_one_(q);
        require_rank_one_(q_dot);
        require_rank_one_(q_ddot);
        return forward_kinematics(joint, xvector<>(q), xvector<>(q_dot), xvector<>(q_ddot));
    }

    // As on jacobian() above, and any one wrong argument is enough.
    template <xexpression_like Q, xexpression_like QD, xexpression_like QDD>
        requires(!(xrank_same_as_or_dynamic<Q, xvector<>> && xrank_same_as_or_dynamic<QD, xvector<>> &&
                   xrank_same_as_or_dynamic<QDD, xvector<>>))
    frame_kinematics forward_kinematics(std::size_t, const Q&, const QD&, const QDD&) const = delete;

    ///
    /// Computes the kinematics of the chain's last frame, which is the point a TCP limit caps.
    ///
    /// Accepts and refuses exactly the arguments the overloads above do.
    ///
    /// @param q (N_actuated,) joint positions, in chain order
    /// @param q_dot (N_actuated,) joint velocities
    /// @param q_ddot (N_actuated,) joint accelerations
    /// @return The last frame's position, linear velocity and linear acceleration
    /// @throws std::invalid_argument as the overloads above do
    ///
    // The arguments are passed through unchanged, and the constraint asks only whether the full
    // form would accept them, so the rules about rank are stated in one place. The constraint must
    // be a trailing requires clause, because the class is still incomplete where a leading one
    // would appear.
    template <typename Q, typename QD, typename QDD>
    [[nodiscard]] frame_kinematics forward_kinematics(const Q& q, const QD& q_dot, const QDD& q_ddot) const
        requires requires(const Q& qq, const QD& qqd, const QDD& qqdd) { this->forward_kinematics(std::size_t{0}, qq, qqd, qqdd); }
    {
        return forward_kinematics(size() - 1, q, q_dot, q_ddot);
    }

    ///
    /// Computes the position of the chain's last frame at joint positions q.
    ///
    /// Trajex does not need this itself. It exists so that a caller can compare the chain it built
    /// against its own forward kinematics.
    ///
    /// @param q (N_actuated,) joint positions, in chain order
    /// @return The position, in the length unit of the chain's translations
    /// @throws std::invalid_argument on q-size mismatch, or if q is not 1-dimensional
    ///
    template <typename Q>
    [[nodiscard]] std::array<double, 3> end_position(const Q& q) const
        requires requires(const Q& qq) { this->forward_kinematics(qq, qq, qq); }
    {
        const xvector<> at_rest = xt::zeros<double>({dof()});
        return forward_kinematics(q, at_rest, at_rest).position;
    }

   private:
    // A joint is one of three types, and which one says how it moves, as the builder overload
    // that added it does. Each holds what it was given, which visit_joints reports back so that a
    // rebuilt chain is identical to this one, and what build derives from that once: everything
    // the forward-kinematics walk needs that does not depend on q. The walk runs several times per
    // integration step, so none of it is recomputed there. All three are defined after the class,
    // since they hold the nested description types by value and those are not yet complete here.
    struct joint_base_;
    struct fixed_joint_;
    template <typename Axis>
    struct moving_joint_;
    using rotating_joint_ = moving_joint_<rotation_axis>;
    using translating_joint_ = moving_joint_<translation_axis>;
    using joint_ = std::variant<fixed_joint_, rotating_joint_, translating_joint_>;

    // The world-frame axis and origin of each moving joint, and the position of the frame where the
    // forward kinematics stopped.
    struct chain_state_;

    // Takes joints the builder has already validated and derived, and counts the moving ones.
    explicit chain(std::vector<joint_> joints);

    // Defined here, not in the source file: the deduced overloads above instantiate in the
    // caller's translation unit, so a definition there would have to be exported.
    static void require_rank_one_(const auto& value) {
        if (!xrank_is_same_as<xvector<>>(value)) {
            throw std::invalid_argument("viam::trajex::kinematics: expected a 1D joint vector, got " + std::to_string(value.dimension()) +
                                        "D");
        }
    }

    // Evaluates the forward kinematics at joint positions q through joint last_joint, capturing what
    // the Jacobian and forward_kinematics need about each moving joint. Throws std::invalid_argument
    // if q has the wrong size.
    chain_state_ compute_chain_state_(const xvector<>& q, std::size_t last_joint) const;

    std::vector<joint_> joints_;
    std::size_t actuated_count_ = 0;
};

///
/// The rigid transform from a joint's parent frame to the joint's own frame. It applies before
/// the joint's motion.
///
struct chain::joint_origin {
    std::array<double, 3> translation{};

    /// Quaternion w, x, y, z in the Hamilton convention. It need not be unit length, since the
    /// builder normalizes it.
    std::array<double, 4> rotation{1.0, 0.0, 0.0, 0.0};
};

///
/// The axis a revolute or continuous joint rotates about, in the joint's own frame. It need not
/// be unit length.
///
struct chain::rotation_axis {
    std::array<double, 3> direction{};
};

///
/// The axis a prismatic joint translates along, in the joint's own frame. It need not be unit
/// length.
///
struct chain::translation_axis {
    std::array<double, 3> direction{};
};

// What every joint has: its origin as given, and the unit quaternion build derives from the
// origin's rotation. The origin's translation is used as given.
struct chain::joint_base_ {
    joint_origin origin;
    std::array<double, 4> origin_rotation;
};

struct chain::fixed_joint_ : joint_base_ {};

// A rotating or translating joint, which also has its axis as given and the normalized form
// build derives from it.
template <typename Axis>
struct chain::moving_joint_ : joint_base_ {
    Axis axis;
    std::array<double, 3> unit_axis;
};

///
/// Assembles a chain one joint at a time, from the base to the end effector.
///
/// The overload chosen says how each joint moves: an origin alone is a fixed joint, an origin and
/// a rotation_axis a revolute or continuous joint, and an origin and a translation_axis a
/// prismatic joint. Revolute and continuous joints are not distinguished because the chain carries
/// no position limits, and nothing else about them differs. The capped point of a TCP limit is the
/// origin of the last joint's frame, so a chain should end at whatever point the caller wants
/// limited, with a fixed joint if need be.
///
class chain::builder {
   public:
    ///
    /// Appends a fixed joint.
    ///
    /// @param origin Transform from the previous frame
    /// @return This builder
    ///
    builder& extend(const joint_origin& origin) &;

    ///
    /// Appends a revolute or continuous joint.
    ///
    /// @param origin Transform from the previous frame, applied before the rotation
    /// @param axis Axis of rotation in the joint's frame
    /// @return This builder
    ///
    builder& extend(const joint_origin& origin, const rotation_axis& axis) &;

    ///
    /// Appends a prismatic joint.
    ///
    /// @param origin Transform from the previous frame, applied before the translation
    /// @param axis Axis of translation in the joint's frame
    /// @return This builder
    ///
    builder& extend(const joint_origin& origin, const translation_axis& axis) &;

    ///
    /// The overloads above for a builder that is a temporary, so that a chain can be built in a
    /// single expression ending in build().
    ///
    /// @return This builder, as an rvalue
    ///
    builder&& extend(const joint_origin& origin) && {
        return std::move(extend(origin));
    }

    /// @copydoc extend(const joint_origin&) &&
    builder&& extend(const joint_origin& origin, const rotation_axis& axis) && {
        return std::move(extend(origin, axis));
    }

    /// @copydoc extend(const joint_origin&) &&
    builder&& extend(const joint_origin& origin, const translation_axis& axis) && {
        return std::move(extend(origin, axis));
    }

    ///
    /// Validates the joints and builds the chain.
    ///
    /// Quaternions and axes are normalized here, so neither need be unit length on the way in.
    ///
    /// @return The chain
    /// @throws std::invalid_argument naming the offending joint if the chain is empty, or a joint
    ///         has a non-finite translation, a zero or non-finite quaternion, or a zero or
    ///         non-finite axis
    ///
    [[nodiscard]] chain build() &&;

   private:
    // Joints as the caller described them, with their derived values not yet filled in.
    // Validation and derivation wait for build(), which is the one place that can report an error
    // with the joint's index.
    std::vector<joint_> joints_;
};

inline std::size_t chain::size() const noexcept {
    return joints_.size();
}

template <typename Visitor>
void chain::visit_joints(Visitor&& visitor) const {
    for (const joint_& joint : joints_) {
        std::visit(
            [&]<typename Joint>(const Joint& j) {
                if constexpr (std::is_same_v<Joint, fixed_joint_>) {
                    visitor(j.origin);
                } else {
                    visitor(j.origin, j.axis);
                }
            },
            joint);
    }
}

}  // namespace viam::trajex::kinematics
