#include <viam/trajex/jacobian/jacobian.hpp>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <viam/sdk/referenceframe/kinematics_model_table.hpp>

namespace viam::trajex::jacobian {

namespace {

// The SDK's ModelTable carries xyz/rpy/axis as viam::sdk::Vector3 (a thin
// wrapper around std::array<double, 3>, no linalg ops). trajex internally
// uses Eigen, so we adapt at the boundary.
inline Eigen::Vector3d to_eigen(const viam::sdk::Vector3& v) {
    return Eigen::Vector3d(v.x(), v.y(), v.z());
}

// Build the per-link 4x4 transform from URDF (xyz, rpy). URDF rpy is
// fixed-axis XYZ, equivalent to Rz(yaw) * Ry(pitch) * Rx(roll).
inline Eigen::Matrix4d link_transform(const Eigen::Vector3d& xyz,
                                      const Eigen::Vector3d& rpy) {
    const Eigen::Matrix3d R =
        (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = xyz;
    return T;
}

// Validate types and return revolute count. Throws std::invalid_argument
// for unsupported joint types or zero-magnitude revolute axes.
std::size_t validate_and_count_actuated(const viam::sdk::ModelTable& table) {
    std::size_t n_actuated = 0;
    const auto& rows = table.rows();
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        switch (r.type) {
            case viam::sdk::ModelTable::JointType::k_revolute:
                if (to_eigen(r.axis).squaredNorm() < 1e-24) {
                    throw std::invalid_argument(
                        "viam::trajex::jacobian: row " + std::to_string(i) +
                        " is a revolute joint with zero-magnitude axis");
                }
                ++n_actuated;
                break;
            case viam::sdk::ModelTable::JointType::k_fixed:
                break;
            case viam::sdk::ModelTable::JointType::k_continuous:
            case viam::sdk::ModelTable::JointType::k_prismatic:
                throw std::invalid_argument(
                    "viam::trajex::jacobian: row " + std::to_string(i) +
                    " has unsupported joint type (only revolute and fixed are supported)");
        }
    }
    return n_actuated;
}

void check_q_size(std::size_t n_actuated, std::size_t q_size) {
    if (q_size != n_actuated) {
        throw std::invalid_argument(
            "viam::trajex::jacobian: q size mismatch: expected " +
            std::to_string(n_actuated) + " (actuated joints), got " +
            std::to_string(q_size));
    }
}

}  // namespace

xt::xarray<double> compute_jacobian(const xt::xarray<double>& model_table,
                                    const xt::xarray<double>& q) {
    const auto table = viam::sdk::ModelTable::from(model_table);
    const std::size_t n_actuated = validate_and_count_actuated(table);
    check_q_size(n_actuated, q.size());

    // Walk the chain. For each revolute joint, capture its world-frame axis
    // and origin BEFORE applying joint motion (equivalent to post-motion for
    // rotation about own axis; using pre-motion is clearer). After the walk,
    // p_e = translation of final T.
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> per_joint;
    per_joint.reserve(n_actuated);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    std::size_t qi = 0;
    for (const auto& row : table.rows()) {
        T = T * link_transform(to_eigen(row.xyz), to_eigen(row.rpy));

        if (row.type == viam::sdk::ModelTable::JointType::k_revolute) {
            const Eigen::Vector3d axis_local = to_eigen(row.axis).normalized();
            const Eigen::Vector3d w_world = T.block<3, 3>(0, 0) * axis_local;
            const Eigen::Vector3d p_joint = T.block<3, 1>(0, 3);
            per_joint.emplace_back(w_world, p_joint);

            Eigen::Matrix4d T_motion = Eigen::Matrix4d::Identity();
            T_motion.block<3, 3>(0, 0) =
                Eigen::AngleAxisd(q(qi), axis_local).toRotationMatrix();
            T = T * T_motion;
            ++qi;
        }
        // fixed: no motion to apply.
    }

    const Eigen::Vector3d p_e = T.block<3, 1>(0, 3);

    xt::xarray<double> J = xt::zeros<double>({std::size_t{6}, n_actuated});
    for (std::size_t i = 0; i < n_actuated; ++i) {
        const auto& [w, p] = per_joint[i];
        const Eigen::Vector3d Jv = w.cross(p_e - p);
        J(0, i) = Jv.x();
        J(1, i) = Jv.y();
        J(2, i) = Jv.z();
        J(3, i) = w.x();
        J(4, i) = w.y();
        J(5, i) = w.z();
    }
    return J;
}

Eigen::Matrix4d forward_kinematics(const xt::xarray<double>& model_table,
                                   const xt::xarray<double>& q) {
    const auto table = viam::sdk::ModelTable::from(model_table);
    const std::size_t n_actuated = validate_and_count_actuated(table);
    check_q_size(n_actuated, q.size());

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    std::size_t qi = 0;
    for (const auto& row : table.rows()) {
        T = T * link_transform(to_eigen(row.xyz), to_eigen(row.rpy));

        if (row.type == viam::sdk::ModelTable::JointType::k_revolute) {
            const Eigen::Vector3d axis_local = to_eigen(row.axis).normalized();
            Eigen::Matrix4d T_motion = Eigen::Matrix4d::Identity();
            T_motion.block<3, 3>(0, 0) =
                Eigen::AngleAxisd(q(qi), axis_local).toRotationMatrix();
            T = T * T_motion;
            ++qi;
        }
    }
    return T;
}

}  // namespace viam::trajex::jacobian
