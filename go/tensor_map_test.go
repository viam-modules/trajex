//go:build !windows && cgo

package trajex_test

import (
	"testing"

	"go.viam.com/test"

	"github.com/viam-modules/trajex/go"
)

func TestNewCloseRoundTrip(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	test.That(t, m, test.ShouldNotBeNil)
	m.Close()
}

func TestCloseIsNilSafe(t *testing.T) {
	var m *trajex.TensorMap
	m.Close() // must not panic
}

func TestCloseIsIdempotent(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	m.Close()
	m.Close() // must not panic
}

func TestInsertViewFloat64Roundtrip(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	in := []float64{1.5, -2.25, 3.75, 4.0, -5.5, 6.625}
	err = m.InsertFloat64s("x", []uint64{2, 3}, in)
	test.That(t, err, test.ShouldBeNil)

	shape, data, ok, err := m.ViewFloat64s("x")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, shape, test.ShouldResemble, []uint64{2, 3})
	test.That(t, data, test.ShouldResemble, in)
}

func TestInsertViewInt64Roundtrip(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	in := []int64{-1, 0, 1, 2, 3, 4}
	err = m.InsertInt64s("y", []uint64{6}, in)
	test.That(t, err, test.ShouldBeNil)

	shape, data, ok, err := m.ViewInt64s("y")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, shape, test.ShouldResemble, []uint64{6})
	test.That(t, data, test.ShouldResemble, in)
}

func TestInsertScalarFloat64(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	err = m.InsertScalarFloat64("tol", 1.25e-3)
	test.That(t, err, test.ShouldBeNil)

	shape, data, ok, err := m.ViewFloat64s("tol")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, shape, test.ShouldResemble, []uint64{1})
	test.That(t, data, test.ShouldResemble, []float64{1.25e-3})
}

func TestInsertScalarInt64(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	err = m.InsertScalarInt64("freq", 200)
	test.That(t, err, test.ShouldBeNil)

	shape, data, ok, err := m.ViewInt64s("freq")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, shape, test.ShouldResemble, []uint64{1})
	test.That(t, data, test.ShouldResemble, []int64{200})
}

func TestViewMissingKey(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	shape, data, ok, err := m.ViewFloat64s("absent")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeFalse)
	test.That(t, shape, test.ShouldBeNil)
	test.That(t, data, test.ShouldBeNil)

	ishape, idata, iok, ierr := m.ViewInt64s("absent")
	test.That(t, ierr, test.ShouldBeNil)
	test.That(t, iok, test.ShouldBeFalse)
	test.That(t, ishape, test.ShouldBeNil)
	test.That(t, idata, test.ShouldBeNil)
}

func TestViewWrongDtypeIsError(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	err = m.InsertFloat64s("a", []uint64{2}, []float64{1, 2})
	test.That(t, err, test.ShouldBeNil)

	_, _, _, err = m.ViewInt64s("a")
	test.That(t, err, test.ShouldNotBeNil)
}

// Replace-on-duplicate: a second insert under the same key replaces the
// prior contents; views taken after replace see the new data. (The CAPI
// guarantees the prior pointer is invalidated by replace; that's documented
// caller responsibility and not safely testable from Go.)
func TestReplaceOnDuplicate(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	err = m.InsertFloat64s("k", []uint64{3}, []float64{1, 2, 3})
	test.That(t, err, test.ShouldBeNil)

	err = m.InsertFloat64s("k", []uint64{2}, []float64{10, 20})
	test.That(t, err, test.ShouldBeNil)

	shape, data, ok, err := m.ViewFloat64s("k")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, shape, test.ShouldResemble, []uint64{2})
	test.That(t, data, test.ShouldResemble, []float64{10, 20})
}

// Replacing a key may change dtype as well as shape.
func TestReplaceChangesDtype(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	err = m.InsertFloat64s("k", []uint64{2}, []float64{1.5, 2.5})
	test.That(t, err, test.ShouldBeNil)

	err = m.InsertInt64s("k", []uint64{3}, []int64{7, 8, 9})
	test.That(t, err, test.ShouldBeNil)

	_, _, _, ferr := m.ViewFloat64s("k")
	test.That(t, ferr, test.ShouldNotBeNil)

	shape, data, ok, ierr := m.ViewInt64s("k")
	test.That(t, ierr, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, shape, test.ShouldResemble, []uint64{3})
	test.That(t, data, test.ShouldResemble, []int64{7, 8, 9})
}

// A view aliases C-owned storage. Subsequent inserts of *other* keys must
// not invalidate it (std::unordered_map does not invalidate existing
// element pointers on rehash, and the variant slot is stable).
func TestViewStableAcrossUnrelatedInserts(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	err = m.InsertFloat64s("a", []uint64{3}, []float64{1, 2, 3})
	test.That(t, err, test.ShouldBeNil)

	_, dataA, ok, err := m.ViewFloat64s("a")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)

	// Several unrelated inserts that force rehashes.
	for _, k := range []string{"b", "c", "d", "e", "f", "g", "h", "i", "j"} {
		err = m.InsertFloat64s(k, []uint64{1}, []float64{99})
		test.That(t, err, test.ShouldBeNil)
	}

	// dataA is a slice aliasing C-owned storage; must still read the
	// original values.
	test.That(t, dataA, test.ShouldResemble, []float64{1, 2, 3})
}

func TestInsertShapeValidation(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer m.Close()

	// Empty shape rejected.
	err = m.InsertFloat64s("k", nil, []float64{1, 2})
	test.That(t, err, test.ShouldNotBeNil)

	// Zero dim rejected.
	err = m.InsertFloat64s("k", []uint64{2, 0}, []float64{})
	test.That(t, err, test.ShouldNotBeNil)

	// Shape-product / data-length mismatch rejected.
	err = m.InsertFloat64s("k", []uint64{2, 3}, []float64{1, 2, 3, 4, 5})
	test.That(t, err, test.ShouldNotBeNil)
}

func TestUnsafeHandleReflectsClose(t *testing.T) {
	m, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)

	// Use == nil rather than test.ShouldBeNil here: smarty-assertions' nil
	// checker does not handle typed nil unsafe.Pointer values uniformly.
	test.That(t, m.UnsafeHandle() == nil, test.ShouldBeFalse)

	m.Close()
	test.That(t, m.UnsafeHandle() == nil, test.ShouldBeTrue)

	var nilMap *trajex.TensorMap
	test.That(t, nilMap.UnsafeHandle() == nil, test.ShouldBeTrue)
}
