#define BOOST_TEST_MODULE trajex_types_test

#include <chrono>
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if __has_include(<xtensor/containers/xfixed.hpp>)
#include <xtensor/containers/xfixed.hpp>
#else
#include <xtensor/xfixed.hpp>
#endif

#include <viam/trajex/types/arc_acceleration.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/arc_operations.hpp>
#include <viam/trajex/types/arc_velocity.hpp>
#include <viam/trajex/types/epsilon.hpp>
#include <viam/trajex/types/xt.hpp>

#include <boost/test/unit_test.hpp>

using namespace std::chrono_literals;

BOOST_AUTO_TEST_SUITE(arc_length_tests)

BOOST_AUTO_TEST_CASE(arc_length_arithmetic) {
    using namespace viam::trajex;

    const arc_length a{5.0};
    const arc_length b{3.0};

    // Addition
    const arc_length sum = a + b;
    BOOST_CHECK_EQUAL(static_cast<double>(sum), 8.0);

    // Subtraction
    const arc_length diff = a - b;
    BOOST_CHECK_EQUAL(static_cast<double>(diff), 2.0);

    // Scalar multiplication
    const arc_length scaled1 = a * 2.0;
    BOOST_CHECK_EQUAL(static_cast<double>(scaled1), 10.0);

    const arc_length scaled2 = 2.0 * a;
    BOOST_CHECK_EQUAL(static_cast<double>(scaled2), 10.0);

    // Scalar division
    const arc_length divided = a / 2.0;
    BOOST_CHECK_EQUAL(static_cast<double>(divided), 2.5);

    // Compound operations
    arc_length c{10.0};
    c += b;
    BOOST_CHECK_EQUAL(static_cast<double>(c), 13.0);

    c -= arc_length{3.0};
    BOOST_CHECK_EQUAL(static_cast<double>(c), 10.0);

    c *= 2.0;
    BOOST_CHECK_EQUAL(static_cast<double>(c), 20.0);

    c /= 4.0;
    BOOST_CHECK_EQUAL(static_cast<double>(c), 5.0);
}

BOOST_AUTO_TEST_CASE(arc_length_comparison) {
    using namespace viam::trajex;

    const arc_length a{5.0};
    const arc_length b{3.0};
    const arc_length c{5.0};

    BOOST_CHECK(a > b);
    BOOST_CHECK(b < a);
    BOOST_CHECK(a >= c);
    BOOST_CHECK(a <= c);
    BOOST_CHECK(a == c);
    BOOST_CHECK(a != b);
}

BOOST_AUTO_TEST_CASE(arc_length_explicit_conversion) {
    using namespace viam::trajex;

    const arc_length len{5.5};

    // Explicit conversion should work
    const double value = static_cast<double>(len);
    BOOST_CHECK_EQUAL(value, 5.5);

    // Direct initialization should work
    const double value2{len};
    BOOST_CHECK_EQUAL(value2, 5.5);
}

BOOST_AUTO_TEST_CASE(arc_length_assignment) {
    using namespace viam::trajex;

    arc_length a{5.0};
    const arc_length b{10.0};

    // Same-type assignment should work
    a = b;
    BOOST_CHECK_EQUAL(static_cast<double>(a), 10.0);

    // Assignment from constructed value should work
    a = arc_length{7.5};
    BOOST_CHECK_EQUAL(static_cast<double>(a), 7.5);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(arc_velocity_tests)

BOOST_AUTO_TEST_CASE(arc_velocity_arithmetic) {
    using namespace viam::trajex;

    const arc_velocity v1{10.0};
    const arc_velocity v2{3.0};

    // Addition
    const arc_velocity sum = v1 + v2;
    BOOST_CHECK_EQUAL(static_cast<double>(sum), 13.0);

    // Subtraction
    const arc_velocity diff = v1 - v2;
    BOOST_CHECK_EQUAL(static_cast<double>(diff), 7.0);

    // Scalar multiplication
    const arc_velocity scaled1 = v1 * 2.0;
    BOOST_CHECK_EQUAL(static_cast<double>(scaled1), 20.0);

    const arc_velocity scaled2 = 2.0 * v1;
    BOOST_CHECK_EQUAL(static_cast<double>(scaled2), 20.0);

    // Scalar division
    const arc_velocity divided = v1 / 2.0;
    BOOST_CHECK_EQUAL(static_cast<double>(divided), 5.0);

    // Self-division (ratio)
    const double ratio = v1 / v2;
    BOOST_CHECK_CLOSE(ratio, 10.0 / 3.0, 0.0001);

    // Compound operations
    arc_velocity v3{15.0};
    v3 += v2;
    BOOST_CHECK_EQUAL(static_cast<double>(v3), 18.0);

    v3 -= arc_velocity{8.0};
    BOOST_CHECK_EQUAL(static_cast<double>(v3), 10.0);

    v3 *= 3.0;
    BOOST_CHECK_EQUAL(static_cast<double>(v3), 30.0);

    v3 /= 5.0;
    BOOST_CHECK_EQUAL(static_cast<double>(v3), 6.0);
}

BOOST_AUTO_TEST_CASE(arc_velocity_comparison) {
    using namespace viam::trajex;

    const arc_velocity v1{10.0};
    const arc_velocity v2{5.0};
    const arc_velocity v3{10.0};

    BOOST_CHECK(v1 > v2);
    BOOST_CHECK(v2 < v1);
    BOOST_CHECK(v1 >= v3);
    BOOST_CHECK(v1 <= v3);
    BOOST_CHECK(v1 == v3);
    BOOST_CHECK(v1 != v2);
}

BOOST_AUTO_TEST_CASE(arc_velocity_explicit_conversion) {
    using namespace viam::trajex;

    const arc_velocity vel{7.5};

    // Explicit conversion should work
    const double value = static_cast<double>(vel);
    BOOST_CHECK_EQUAL(value, 7.5);
}

BOOST_AUTO_TEST_CASE(arc_velocity_assignment) {
    using namespace viam::trajex;

    arc_velocity v1{5.0};
    const arc_velocity v2{12.5};

    // Same-type assignment should work
    v1 = v2;
    BOOST_CHECK_EQUAL(static_cast<double>(v1), 12.5);

    // Assignment from constructed value should work
    v1 = arc_velocity{8.0};
    BOOST_CHECK_EQUAL(static_cast<double>(v1), 8.0);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(arc_acceleration_tests)

BOOST_AUTO_TEST_CASE(arc_acceleration_arithmetic) {
    using namespace viam::trajex;

    const arc_acceleration a1{8.0};
    const arc_acceleration a2{2.0};

    // Addition
    const arc_acceleration sum = a1 + a2;
    BOOST_CHECK_EQUAL(static_cast<double>(sum), 10.0);

    // Subtraction
    const arc_acceleration diff = a1 - a2;
    BOOST_CHECK_EQUAL(static_cast<double>(diff), 6.0);

    // Scalar multiplication
    const arc_acceleration scaled1 = a1 * 3.0;
    BOOST_CHECK_EQUAL(static_cast<double>(scaled1), 24.0);

    const arc_acceleration scaled2 = 3.0 * a1;
    BOOST_CHECK_EQUAL(static_cast<double>(scaled2), 24.0);

    // Scalar division
    const arc_acceleration divided = a1 / 4.0;
    BOOST_CHECK_EQUAL(static_cast<double>(divided), 2.0);

    // Self-division (ratio)
    const double ratio = a1 / a2;
    BOOST_CHECK_EQUAL(ratio, 4.0);

    // Compound operations
    arc_acceleration a3{12.0};
    a3 += a2;
    BOOST_CHECK_EQUAL(static_cast<double>(a3), 14.0);

    a3 -= arc_acceleration{4.0};
    BOOST_CHECK_EQUAL(static_cast<double>(a3), 10.0);

    a3 *= 2.0;
    BOOST_CHECK_EQUAL(static_cast<double>(a3), 20.0);

    a3 /= 5.0;
    BOOST_CHECK_EQUAL(static_cast<double>(a3), 4.0);
}

BOOST_AUTO_TEST_CASE(arc_acceleration_comparison) {
    using namespace viam::trajex;

    const arc_acceleration a1{8.0};
    const arc_acceleration a2{3.0};
    const arc_acceleration a3{8.0};

    BOOST_CHECK(a1 > a2);
    BOOST_CHECK(a2 < a1);
    BOOST_CHECK(a1 >= a3);
    BOOST_CHECK(a1 <= a3);
    BOOST_CHECK(a1 == a3);
    BOOST_CHECK(a1 != a2);
}

BOOST_AUTO_TEST_CASE(arc_acceleration_explicit_conversion) {
    using namespace viam::trajex;

    const arc_acceleration acc{4.5};

    // Explicit conversion should work
    const double value = static_cast<double>(acc);
    BOOST_CHECK_EQUAL(value, 4.5);
}

BOOST_AUTO_TEST_CASE(arc_acceleration_assignment) {
    using namespace viam::trajex;

    arc_acceleration a1{3.0};
    const arc_acceleration a2{9.5};

    // Same-type assignment should work
    a1 = a2;
    BOOST_CHECK_EQUAL(static_cast<double>(a1), 9.5);

    // Assignment from constructed value should work
    a1 = arc_acceleration{6.0};
    BOOST_CHECK_EQUAL(static_cast<double>(a1), 6.0);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(arc_dimensional_analysis_tests)

BOOST_AUTO_TEST_CASE(velocity_times_duration_yields_length) {
    using namespace viam::trajex;

    const arc_velocity vel{10.0};  // ds/dt
    const auto dt = 0.5s;          // dt

    // velocity * time = length
    const arc_length len = vel * dt;
    BOOST_CHECK_EQUAL(static_cast<double>(len), 5.0);
}

BOOST_AUTO_TEST_CASE(velocity_divided_by_duration_yields_acceleration) {
    using namespace viam::trajex;

    const arc_velocity vel{10.0};  // ds/dt
    const auto dt = 2.0s;          // dt

    // velocity / time = acceleration
    const arc_acceleration acc = vel / dt;
    BOOST_CHECK_EQUAL(static_cast<double>(acc), 5.0);
}

BOOST_AUTO_TEST_CASE(acceleration_times_duration_yields_velocity) {
    using namespace viam::trajex;

    const arc_acceleration acc{5.0};  // d^2s/dt^2
    const auto dt = 2.0s;             // dt

    // acceleration * time = velocity
    const arc_velocity vel = acc * dt;
    BOOST_CHECK_EQUAL(static_cast<double>(vel), 10.0);
}

BOOST_AUTO_TEST_CASE(length_divided_by_duration_yields_velocity) {
    using namespace viam::trajex;

    const arc_length len{15.0};  // ds
    const auto dt = 3.0s;        // dt

    // length / time = velocity
    const arc_velocity vel = len / dt;
    BOOST_CHECK_EQUAL(static_cast<double>(vel), 5.0);
}

BOOST_AUTO_TEST_CASE(velocity_divided_by_acceleration_yields_duration) {
    using namespace viam::trajex;

    const arc_velocity vel{20.0};     // ds/dt
    const arc_acceleration acc{5.0};  // d^2s/dt^2

    // velocity / acceleration = time
    const auto dt = vel / acc;
    BOOST_CHECK_EQUAL(dt.count(), 4.0);
}

BOOST_AUTO_TEST_CASE(calculus_operations_are_unidirectional) {
    using namespace viam::trajex;

    // These operations should compile (time on RHS):
    const arc_velocity vel{10.0};
    const arc_acceleration acc{5.0};
    const arc_length len{15.0};
    const auto dt = 2.0s;

    // velocity * duration -> length (OK)
    [[maybe_unused]] const arc_length l1 = vel * dt;

    // acceleration * duration -> velocity (OK)
    [[maybe_unused]] const arc_velocity v1 = acc * dt;

    // length / duration -> velocity (OK)
    [[maybe_unused]] const arc_velocity v2 = len / dt;

    // velocity / duration -> acceleration (OK)
    [[maybe_unused]] const arc_acceleration a1 = vel / dt;

    // The following should NOT compile (commented out to allow test to build):
    // [[maybe_unused]] const arc_length l2 = dt * vel;  // duration * velocity - NOT ALLOWED
    // [[maybe_unused]] const arc_velocity v3 = dt * acc;  // duration * acceleration - NOT ALLOWED
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(arc_constexpr_tests)

BOOST_AUTO_TEST_CASE(constexpr_operations) {
    using namespace viam::trajex;

    // Verify compile-time evaluation of cross-type operations
    constexpr arc_velocity vel{10.0};
    constexpr auto dt = std::chrono::duration<double>{0.5};
    constexpr arc_length len = vel * dt;
    static_assert(static_cast<double>(len) == 5.0, "velocity * duration should be constexpr");

    constexpr arc_acceleration acc{4.0};
    constexpr arc_velocity vel2 = acc * dt;
    static_assert(static_cast<double>(vel2) == 2.0, "acceleration * duration should be constexpr");

    constexpr arc_velocity vel3{20.0};
    constexpr arc_acceleration acc2 = vel3 / dt;
    static_assert(static_cast<double>(acc2) == 40.0, "velocity / duration should be constexpr");

    constexpr arc_length len2{12.0};
    constexpr arc_velocity vel4 = len2 / dt;
    static_assert(static_cast<double>(vel4) == 24.0, "length / duration should be constexpr");

    // Verify compile-time evaluation of same-type operations
    constexpr arc_length len3 = arc_length{5.0} + arc_length{3.0};
    static_assert(static_cast<double>(len3) == 8.0, "arc_length addition should be constexpr");

    constexpr arc_velocity vel5 = arc_velocity{10.0} - arc_velocity{3.0};
    static_assert(static_cast<double>(vel5) == 7.0, "arc_velocity subtraction should be constexpr");

    constexpr arc_acceleration acc3 = arc_acceleration{6.0} * 2.0;
    static_assert(static_cast<double>(acc3) == 12.0, "arc_acceleration scalar multiplication should be constexpr");
}

BOOST_AUTO_TEST_CASE(epsilon_comparisons) {
    using namespace viam::trajex;

    const epsilon eps{1e-9};

    // Test comparisons with arc_length
    const arc_length len1{1e-10};
    const arc_length len2{1e-8};
    BOOST_CHECK(len1 < eps);
    BOOST_CHECK(len2 > eps);
    BOOST_CHECK(arc_length{1e-9} == eps);

    // Test comparisons with arc_velocity
    const arc_velocity vel1{5e-10};
    const arc_velocity vel2{5e-8};
    BOOST_CHECK(vel1 < eps);
    BOOST_CHECK(vel2 > eps);

    // Test comparisons with arc_acceleration
    const arc_acceleration acc1{2e-10};
    const arc_acceleration acc2{2e-8};
    BOOST_CHECK(acc1 < eps);
    BOOST_CHECK(acc2 > eps);

    // Test with abs() for typical use case
    const arc_velocity v1{1.0};
    const arc_velocity v2{1.0 + 1e-10};
    BOOST_CHECK(abs(v1 - v2) < eps);

    // Test epsilon scaling
    const epsilon eps2 = eps * 10.0;
    BOOST_CHECK(static_cast<double>(eps2) == 1e-8);

    const epsilon eps3 = 5.0 * eps;
    BOOST_CHECK(static_cast<double>(eps3) == 5e-9);

    const epsilon eps4 = eps / 2.0;
    BOOST_CHECK(static_cast<double>(eps4) == 5e-10);

    // Test compound assignment
    epsilon eps5{2.0};
    eps5 *= 4.0;
    BOOST_CHECK_EQUAL(static_cast<double>(eps5), 8.0);

    eps5 /= 2.0;
    BOOST_CHECK_EQUAL(static_cast<double>(eps5), 4.0);

    // Verify epsilon only allows comparison and scaling, not addition/subtraction
    // The following should NOT compile:
    // auto bad = eps + eps;  // Should fail
    // auto bad2 = len1 + eps;  // Should fail
    // auto bad3 = eps - eps;  // Should fail
}

BOOST_AUTO_TEST_CASE(epsilon_wrapper_less_than) {
    using namespace viam::trajex;

    const epsilon eps{1.0};

    // Significantly different values
    const arc_length a{5.0};
    const arc_length b{10.0};
    BOOST_CHECK(eps.wrap(a) < eps.wrap(b));     // 5 < 10 significantly
    BOOST_CHECK(!(eps.wrap(b) < eps.wrap(a)));  // 10 not < 5

    // Within tolerance (not significantly different)
    const arc_length c{10.0};
    const arc_length d{10.5};
    BOOST_CHECK(!(eps.wrap(c) < eps.wrap(d)));  // 10 not < 10.5 (within eps)
    BOOST_CHECK(!(eps.wrap(d) < eps.wrap(c)));  // 10.5 not < 10 (within eps)

    // Right at the boundary
    const arc_length e{10.0};
    const arc_length f{11.0};
    BOOST_CHECK(!(eps.wrap(e) < eps.wrap(f)));  // Exactly at epsilon boundary

    const arc_length g{10.0};
    const arc_length h{11.1};
    BOOST_CHECK(eps.wrap(g) < eps.wrap(h));  // Just beyond epsilon
}

BOOST_AUTO_TEST_CASE(epsilon_wrapper_greater_than) {
    using namespace viam::trajex;

    const epsilon eps{1.0};

    // Significantly different values
    const arc_velocity v1{20.0};
    const arc_velocity v2{10.0};
    BOOST_CHECK(eps.wrap(v1) > eps.wrap(v2));     // 20 > 10 significantly
    BOOST_CHECK(!(eps.wrap(v2) > eps.wrap(v1)));  // 10 not > 20

    // Within tolerance
    const arc_velocity v3{15.0};
    const arc_velocity v4{15.8};
    BOOST_CHECK(!(eps.wrap(v3) > eps.wrap(v4)));  // Not significantly different
    BOOST_CHECK(!(eps.wrap(v4) > eps.wrap(v3)));  // Not significantly different
}

BOOST_AUTO_TEST_CASE(epsilon_wrapper_equality) {
    using namespace viam::trajex;

    const epsilon eps{1.0};

    // Exactly equal
    const arc_acceleration a1{5.0};
    const arc_acceleration a2{5.0};
    BOOST_CHECK(eps.wrap(a1) == eps.wrap(a2));

    // Within tolerance (indistinguishable)
    const arc_acceleration a3{10.0};
    const arc_acceleration a4{10.5};
    BOOST_CHECK(eps.wrap(a3) == eps.wrap(a4));

    // Just at boundary
    const arc_acceleration a5{10.0};
    const arc_acceleration a6{11.0};
    BOOST_CHECK(eps.wrap(a5) == eps.wrap(a6));

    // Significantly different
    const arc_acceleration a7{10.0};
    const arc_acceleration a8{12.0};
    BOOST_CHECK(!(eps.wrap(a7) == eps.wrap(a8)));
    BOOST_CHECK(eps.wrap(a7) != eps.wrap(a8));
}

BOOST_AUTO_TEST_CASE(epsilon_wrapper_less_equal_greater_equal) {
    using namespace viam::trajex;

    const epsilon eps{1.0};

    const arc_length a{5.0};
    const arc_length b{10.0};
    const arc_length c{10.5};

    // <= means "not significantly greater"
    BOOST_CHECK(eps.wrap(a) <= eps.wrap(b));     // 5 <= 10 (less)
    BOOST_CHECK(eps.wrap(c) <= eps.wrap(b));     // 10.5 <= 10 (equivalent)
    BOOST_CHECK(!(eps.wrap(b) <= eps.wrap(a)));  // 10 not <= 5

    // >= means "not significantly less"
    BOOST_CHECK(eps.wrap(b) >= eps.wrap(a));     // 10 >= 5 (greater)
    BOOST_CHECK(eps.wrap(b) >= eps.wrap(c));     // 10 >= 10.5 (equivalent)
    BOOST_CHECK(!(eps.wrap(a) >= eps.wrap(b)));  // 5 not >= 10
}

BOOST_AUTO_TEST_CASE(epsilon_wrapper_mixed_tolerances) {
    using namespace viam::trajex;

    const epsilon eps_strict{0.5};
    const epsilon eps_loose{2.0};

    const arc_length a{10.0};
    const arc_length b{11.0};

    // Using stricter tolerance, values are different
    BOOST_CHECK(eps_strict.wrap(a) != eps_strict.wrap(b));
    BOOST_CHECK(eps_strict.wrap(a) < eps_strict.wrap(b));

    // Using looser tolerance, values are equivalent
    BOOST_CHECK(eps_loose.wrap(a) == eps_loose.wrap(b));

    // Mixed: should use min (stricter) tolerance
    BOOST_CHECK(eps_strict.wrap(a) != eps_loose.wrap(b));
    BOOST_CHECK(eps_loose.wrap(a) != eps_strict.wrap(b));
    BOOST_CHECK(eps_strict.wrap(a) < eps_loose.wrap(b));
}

BOOST_AUTO_TEST_CASE(epsilon_wrapper_negative_values) {
    using namespace viam::trajex;

    const epsilon eps{1.0};

    const arc_velocity v1{-10.0};
    const arc_velocity v2{-5.0};
    const arc_velocity v3{5.0};

    BOOST_CHECK(eps.wrap(v1) < eps.wrap(v2));  // -10 < -5 significantly
    BOOST_CHECK(eps.wrap(v1) < eps.wrap(v3));  // -10 < 5 significantly
    BOOST_CHECK(eps.wrap(v2) < eps.wrap(v3));  // -5 < 5 significantly

    // Within tolerance around zero
    const arc_velocity v4{0.0};
    const arc_velocity v5{0.5};
    const arc_velocity v6{-0.5};
    BOOST_CHECK(eps.wrap(v4) == eps.wrap(v5));  // 0 == 0.5 within tolerance
    BOOST_CHECK(eps.wrap(v4) == eps.wrap(v6));  // 0 == -0.5 within tolerance
    BOOST_CHECK(eps.wrap(v5) == eps.wrap(v6));  // 0.5 == -0.5 within tolerance
}

BOOST_AUTO_TEST_CASE(epsilon_wrapper_constexpr) {
    using namespace viam::trajex;

    // Verify wrapper operations can be constexpr
    static constexpr epsilon eps{1.0};
    static constexpr arc_length a{5.0};
    static constexpr arc_length b{10.0};

    // These should all be constexpr evaluable
    constexpr auto w1 = eps.wrap(a);
    constexpr auto w2 = eps.wrap(b);

    static_assert((w1 <=> w2) == std::weak_ordering::less, "constexpr comparison should work");
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(xrank_checked_tests)

BOOST_AUTO_TEST_CASE(construction_checks_every_way_in) {
    using namespace viam::trajex;

    const auto holdable = []<typename Held, typename From>() { return std::constructible_from<Held, const From&>; };

    using vec = xrank_checked<xvector<>>;
    using mat = xrank_checked<xmatrix<>>;
    using fixed_two = xt::xtensor_fixed<double, xt::xshape<2, 3>>;
    using rank_three = xt::xtensor<double, 3>;

    // Right rank, nothing to check.
    BOOST_CHECK((holdable.template operator()<vec, xvector<>>()));
    BOOST_CHECK((holdable.template operator()<mat, xmatrix<>>()));
    BOOST_CHECK((holdable.template operator()<mat, fixed_two>()));

    // Rank known only at runtime: admitted here, checked at construction below.
    BOOST_CHECK((holdable.template operator()<vec, xt::xarray<double>>()));
    BOOST_CHECK((holdable.template operator()<mat, xt::xarray<double>>()));

    // Fixed at the wrong rank: refused outright.
    BOOST_CHECK(!(holdable.template operator()<vec, xmatrix<>>()));
    BOOST_CHECK(!(holdable.template operator()<mat, xvector<>>()));
    BOOST_CHECK(!(holdable.template operator()<mat, rank_three>()));

    const xt::xarray<double> two_d = {{1.0, 2.0}, {3.0, 4.0}};
    const xt::xarray<double> one_d = {1.0, 2.0};

    BOOST_CHECK_NO_THROW(static_cast<void>(mat{two_d}));
    BOOST_CHECK_NO_THROW(static_cast<void>(vec{one_d}));
    BOOST_CHECK_THROW(static_cast<void>(mat{one_d}), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(vec{two_d}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(reading_is_transparent_but_not_from_a_temporary) {
    using namespace viam::trajex;

    const xrank_checked<xvector<>> limits = xvector<>{1.0, 2.0, 3.0};

    // Passing it along needs no ceremony.
    const auto sum = [](const xvector<>& v) { return v(0) + v(1) + v(2); };
    BOOST_CHECK_CLOSE(sum(limits), 6.0, 1e-12);

    // Reaching inside does, and reads the same value.
    BOOST_CHECK_EQUAL(limits->size(), 3U);
    BOOST_CHECK_CLOSE((*limits)(1), 2.0, 1e-12);
    BOOST_CHECK_CLOSE(limits.get()(2), 3.0, 1e-12);

    // A default-constructed one is empty rather than absent, and still rank 1.
    const xrank_checked<xvector<>> unset;
    BOOST_CHECK_EQUAL(unset->dimension(), 1U);
    BOOST_CHECK_EQUAL(unset->size(), 0U);

    // Reaching inside a temporary would hand back a reference to storage that dies at the
    // semicolon, so every way of reading refuses an rvalue, the conversion included.
    using held = xrank_checked<xvector<>>;
    const auto derefs = []<typename T>() { return requires(T&& t) { *std::forward<T>(t); }; };
    const auto arrows = []<typename T>() { return requires(T&& t) { std::forward<T>(t).operator->(); }; };
    const auto gets = []<typename T>() { return requires(T&& t) { std::forward<T>(t).get(); }; };

    BOOST_CHECK(derefs.template operator()<held&>());
    BOOST_CHECK(arrows.template operator()<held&>());
    BOOST_CHECK(gets.template operator()<held&>());

    BOOST_CHECK(!derefs.template operator()<held>());
    BOOST_CHECK(!arrows.template operator()<held>());
    BOOST_CHECK(!gets.template operator()<held>());

    BOOST_CHECK((std::convertible_to<held&, const xvector<>&>));
    BOOST_CHECK(!(std::convertible_to<held, const xvector<>&>));
}

// Whether an rvalue source has anything to consume is xtensor's decision and differs by type,
// so these compare the address of the storage before and after rather than assert a rule.
BOOST_AUTO_TEST_CASE(construction_consumes_an_rvalue_and_spares_an_lvalue) {
    using namespace viam::trajex;

    // The one that matters. Converting a dynamically ranked array steals its buffer, and
    // binding the argument to a const reference on the way in would copy the whole thing.
    {
        xt::xarray<double> source = {{1.0, 2.0}, {3.0, 4.0}};
        const double* const storage = source.data();
        const xrank_checked<xmatrix<>> held{std::move(source)};
        BOOST_CHECK_EQUAL(held.get().data(), storage);
    }

    // An lvalue must be left alone, whatever the rank path.
    {
        const xt::xarray<double> source = {{1.0, 2.0}, {3.0, 4.0}};
        const double* const storage = source.data();
        const xrank_checked<xmatrix<>> held{source};
        BOOST_CHECK_NE(held.get().data(), storage);
        BOOST_CHECK_EQUAL(source.dimension(), 2U);
        BOOST_CHECK_CLOSE(source(1, 1), 4.0, 1e-12);
    }

    // An rvalue already of the held type goes through the dedicated T&& constructor.
    {
        xvector<> source = {1.0, 2.0, 3.0};
        const double* const storage = source.data();
        const xrank_checked<xvector<>> held{std::move(source)};
        BOOST_CHECK_EQUAL(held.get().data(), storage);
    }

    // Consuming the argument must not cost the check: a rank that turns out wrong still throws.
    // The source is built inside the lambda rather than moved from one the macro can see,
    // because BOOST_CHECK_THROW may evaluate its statement twice and a moved-from array would
    // not fail the same way the second time.
    {
        const auto from_a_moved_wrong_rank = [] {
            xt::xarray<double> wrong = {1.0, 2.0, 3.0};
            return xrank_checked<xmatrix<>>{std::move(wrong)};
        };
        BOOST_CHECK_THROW(static_cast<void>(from_a_moved_wrong_rank()), std::invalid_argument);
    }
}

// A forwarding constructor in a copyable class is the classic way to displace the copy
// constructor. It can't happen here, because both require an xtensor expression and
// xrank_checked isn't one, but removing that constraint would look harmless.
BOOST_AUTO_TEST_CASE(forwarding_constructors_do_not_displace_copy_or_move) {
    using namespace viam::trajex;

    using held = xrank_checked<xvector<>>;

    static_assert(std::is_copy_constructible_v<held>);
    static_assert(std::is_move_constructible_v<held>);

    held original{xvector<>{1.0, 2.0, 3.0}};

    const held copied{original};
    BOOST_CHECK_EQUAL(copied.get().size(), 3U);
    BOOST_CHECK_CLOSE(copied.get()(0), 1.0, 1e-12);
    BOOST_CHECK_EQUAL(original.get().size(), 3U);

    const double* const storage = original.get().data();
    const held moved{std::move(original)};
    BOOST_CHECK_EQUAL(moved.get().data(), storage);
}

BOOST_AUTO_TEST_SUITE_END()
