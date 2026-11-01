//go:build !windows && cgo

// Package viamapi turns the kinematic messages of the Viam API, KinematicModel
// and FrameSystemConfig, into trajex joints. It imports go.viam.com/api for the
// messages and nothing from RDK, so a caller holding a GetKinematics or
// FrameSystemConfig reply can build a chain without RDK's frame system. The
// go/rdk package goes through it for a model RDK has already loaded.
package viamapi

import (
	"math"
	"slices"
	"sort"

	"github.com/pkg/errors"
	commonpb "go.viam.com/api/common/v1"
	robotpb "go.viam.com/api/robot/v1"

	"github.com/viam-modules/trajex/go/kinematics"
)

// world names the frame every model and every frame system hangs from. A link
// or joint whose parent is world, or whose parent is not part of the model at
// all, is a root of the model, which is how RDK reads the message too.
const world = "world"

// orientationVectorPoleRadius is how close to straight up or down an
// orientation vector may point before RDK measures its theta differently. The
// value is RDK's, and has to stay so for both sides to read one message the
// same way.
const orientationVectorPoleRadius = 0.0001

// JointsFromKinematicModel describes the serial chain from the base of model
// out to outputFrame, which is where a TCP speed cap applies. The result is
// meant for kinematics.NewChain and always ends in a fixed joint, so a chain
// has the same shape whether or not links follow the last moving joint.
//
// outputFrame names a link or joint of the model. When it is empty the
// model's own output frame is used, which is the first of its output_frames
// or else its single leaf, the rule RDK applies when it loads the model. A
// caller capping a point past that frame, such as the tip of a tool on a
// flange, names the link that stands for the point, or appends a
// kinematics.FixedJoint to the result. It returns an error if outputFrame
// names no link or joint of model, and for a model that is nil or empty.
//
// Lengths are in millimetres, as they are in the message, so prismatic joint
// positions and a TCP speed cap for the chain are in millimetres as well.
// Only geometry is mapped. The limits and properties the message carries are
// left to the caller's policy.
//
// RDK drives a model with one position for every moving joint in the whole
// model, in its own order, and a caller hands trajex those same positions. So
// the moving joints on the path to outputFrame must be exactly the model's
// moving joints in that order, and a model that fails that is refused rather
// than given a chain that would read the positions wrongly. That covers a
// mimic joint on the path, which has no position of its own, and a moving
// joint off the path, such as a gripper's finger.
func JointsFromKinematicModel(model *commonpb.KinematicModel, outputFrame string) ([]kinematics.Joint, error) {
	b := &chainBuilder{origin: identityPose}
	if err := b.appendModel(model, model.GetName(), outputFrame); err != nil {
		return nil, err
	}
	return b.finish(), nil
}

// JointsFromFrameSystemConfig describes the serial chain from the world frame
// of a machine out to the output frame of the part named tip, through every
// part between. It is for a caller holding a machine's FrameSystemConfig
// reply rather than one component's model, such as one capping the speed of a
// gripper carried by an arm that itself rides a gantry.
//
// Each part contributes the fixed transform that places its frame on its
// parent, then the joints of its model, built as JointsFromKinematicModel
// builds them with the model's own output frame. A part with no model is a
// fixed transform alone. Positions for the chain are the positions of each
// part's model in turn, from the part nearest world out to tip, an order of
// this function's own, since a frame system keys positions by part rather
// than listing them. Parts off that path contribute nothing and take no
// positions.
//
// A part whose parent is world, or that names no parent, hangs from the root.
// Any other parent has to be a part of the same reply, since the message
// carries no other frames.
func JointsFromFrameSystemConfig(parts []*robotpb.FrameSystemConfig, tip string) ([]kinematics.Joint, error) {
	if tip == "" {
		return nil, errors.New("trajex/viamapi: the tip of the chain is not named")
	}
	byName := map[string]*robotpb.FrameSystemConfig{}
	for i, part := range parts {
		name := part.GetFrame().GetReferenceFrame()
		if name == "" {
			return nil, errors.Errorf("trajex/viamapi: part %d of the frame system has no frame name", i)
		}
		if _, dup := byName[name]; dup {
			return nil, errors.Errorf("trajex/viamapi: the frame system names part %q more than once", name)
		}
		byName[name] = part
	}
	if _, ok := byName[tip]; !ok {
		return nil, errors.Errorf("trajex/viamapi: the frame system has no part %q", tip)
	}

	var path []*robotpb.FrameSystemConfig
	seen := map[string]bool{}
	for name, child := tip, ""; name != "" && name != world; {
		part, ok := byName[name]
		if !ok {
			return nil, errors.Errorf("trajex/viamapi: part %q hangs from %q, which is not a part of the frame system", child, name)
		}
		if seen[name] {
			return nil, errors.Errorf("trajex/viamapi: the path from %q to the world frame is a cycle", tip)
		}
		seen[name] = true
		path = append(path, part)
		name, child = part.GetFrame().GetPoseInObserverFrame().GetReferenceFrame(), name
	}
	slices.Reverse(path)

	b := &chainBuilder{origin: identityPose}
	for _, part := range path {
		b.fixed(poseFromProto(part.GetFrame().GetPoseInObserverFrame().GetPose()))
		model := part.GetModel()
		if model == nil {
			continue
		}
		label := model.GetName()
		if label == "" {
			label = part.GetFrame().GetReferenceFrame()
		}
		if err := b.appendModel(model, label, ""); err != nil {
			return nil, err
		}
	}
	return b.finish(), nil
}

// chainBuilder collects joints base first. A trajex joint carries the static
// transform that precedes its motion as its origin, while the message gives
// links as static poses of their own, so we compose the links between one
// joint and the next into origin and hand it to the next joint.
type chainBuilder struct {
	joints []kinematics.Joint
	origin pose
}

func (b *chainBuilder) fixed(p pose) {
	b.origin = compose(b.origin, p)
}

func (b *chainBuilder) rotational(axis [3]float64) {
	b.joints = append(b.joints, kinematics.RotationalJoint(b.origin.xyz, b.origin.wxyz, axis))
	b.origin = identityPose
}

func (b *chainBuilder) translational(axis [3]float64) {
	b.joints = append(b.joints, kinematics.TranslationalJoint(b.origin.xyz, b.origin.wxyz, axis))
	b.origin = identityPose
}

func (b *chainBuilder) finish() []kinematics.Joint {
	return append(b.joints, kinematics.FixedJoint(b.origin.xyz, b.origin.wxyz))
}

// appendModel walks model from its root to outputFrame, adding what it finds
// to the chain. label names the model in errors, since a model served inside
// a frame system part may carry no name of its own.
func (b *chainBuilder) appendModel(model *commonpb.KinematicModel, label, outputFrame string) error {
	elements, err := elementsOf(model, label)
	if err != nil {
		return err
	}
	output, err := outputFrameOf(model, label, elements, outputFrame)
	if err != nil {
		return err
	}
	path, err := pathToRoot(elements, output)
	if err != nil {
		return err
	}

	var moving []string
	for _, id := range path {
		e := elements[id]
		if !e.isJoint {
			b.fixed(e.pose)
			continue
		}
		if e.mimic {
			return errors.Errorf("trajex/viamapi: joint %q mimics another joint, which trajex cannot express", id)
		}
		if e.axis == [3]float64{} {
			return errors.Errorf("trajex/viamapi: joint %q has no axis", id)
		}
		switch e.jointType {
		case commonpb.JointType_JOINT_TYPE_REVOLUTE, commonpb.JointType_JOINT_TYPE_CONTINUOUS:
			b.rotational(e.axis)
		case commonpb.JointType_JOINT_TYPE_PRISMATIC:
			b.translational(e.axis)
		default:
			return errors.Errorf("trajex/viamapi: joint %q has type %v, which trajex does not support", id, e.jointType)
		}
		moving = append(moving, id)
	}

	want, err := inputOrder(elements, label)
	if err != nil {
		return err
	}
	if !slices.Equal(moving, want) {
		return errors.Errorf(
			"trajex/viamapi: model %q takes inputs for joints %v, but the path to its output frame %q moves joints %v",
			label, want, output, moving)
	}
	return nil
}

// element is one link or joint of a model. A link has a pose and a joint has
// an axis, and either may be the parent of the other.
type element struct {
	parent    string
	isJoint   bool
	pose      pose
	jointType commonpb.JointType
	axis      [3]float64
	mimic     bool
}

func elementsOf(model *commonpb.KinematicModel, label string) (map[string]*element, error) {
	if model == nil {
		return nil, errors.New("trajex/viamapi: kinematic model is nil")
	}
	if len(model.GetLinks()) == 0 && len(model.GetJoints()) == 0 {
		return nil, errors.Errorf("trajex/viamapi: model %q has no links or joints", label)
	}
	elements := map[string]*element{}
	add := func(id string, e *element) error {
		if id == "" {
			return errors.Errorf("trajex/viamapi: model %q has a link or joint with no id", label)
		}
		if _, dup := elements[id]; dup {
			return errors.Errorf("trajex/viamapi: model %q names %q more than once", label, id)
		}
		elements[id] = e
		return nil
	}
	for _, link := range model.GetLinks() {
		p, err := poseFromLinkPose(link.GetPose())
		if err != nil {
			return nil, errors.Wrapf(err, "trajex/viamapi: link %q of model %q", link.GetId(), label)
		}
		if err := add(link.GetId(), &element{parent: link.GetParent(), pose: p}); err != nil {
			return nil, err
		}
	}
	for _, joint := range model.GetJoints() {
		e := &element{
			parent:    joint.GetParent(),
			isJoint:   true,
			jointType: joint.GetType(),
			axis:      vector(joint.GetAxis()),
			mimic:     joint.GetMimic() != nil,
		}
		if err := add(joint.GetId(), e); err != nil {
			return nil, err
		}
	}
	return elements, nil
}

// outputFrameOf picks the frame the chain ends at: the one the caller named,
// else the one the model names, else the model's single leaf.
func outputFrameOf(model *commonpb.KinematicModel, label string, elements map[string]*element, requested string) (string, error) {
	if requested != "" {
		if _, ok := elements[requested]; !ok {
			return "", errors.Errorf("trajex/viamapi: output frame %q is not a link or joint of model %q", requested, label)
		}
		return requested, nil
	}
	if frames := model.GetOutputFrames(); len(frames) > 0 {
		if len(frames) > 1 {
			// RDK refuses to load such a model, so a chain from it would describe nothing RDK drives
			return "", errors.Errorf("trajex/viamapi: model %q names %d output frames %v, which RDK does not support", label, len(frames), frames)
		}
		if _, ok := elements[frames[0]]; !ok {
			return "", errors.Errorf("trajex/viamapi: output frame %q is not a link or joint of model %q", frames[0], label)
		}
		return frames[0], nil
	}
	hasChild := map[string]bool{}
	for _, e := range elements {
		hasChild[e.parent] = true
	}
	var leaves []string
	for id := range elements {
		if !hasChild[id] {
			leaves = append(leaves, id)
		}
	}
	if len(leaves) != 1 {
		sort.Strings(leaves)
		return "", errors.Errorf("trajex/viamapi: model %q names no output frame and has %d leaves %v rather than one", label, len(leaves), leaves)
	}
	return leaves[0], nil
}

// pathToRoot lists the elements from the model's root down to output. The
// root is the first element whose parent is not an element, usually world.
func pathToRoot(elements map[string]*element, output string) ([]string, error) {
	var path []string
	seen := map[string]bool{}
	for id := output; ; {
		e, ok := elements[id]
		if !ok {
			break
		}
		if seen[id] {
			return nil, errors.Errorf("trajex/viamapi: the path from %q to the root of the model is a cycle", output)
		}
		seen[id] = true
		path = append(path, id)
		id = e.parent
	}
	slices.Reverse(path)
	return path, nil
}

// inputOrder lists the joints that take a position, in the order RDK takes
// them: breadth first from world with siblings sorted by id, leaving out
// mimic joints since they follow another joint's position. An element the
// walk never reaches sits on a cycle, which RDK refuses to load, so we refuse
// it too rather than report an input order that leaves it out.
func inputOrder(elements map[string]*element, label string) ([]string, error) {
	childrenOf := map[string][]string{}
	for id, e := range elements {
		parent := e.parent
		if _, ok := elements[parent]; !ok {
			parent = world
		}
		childrenOf[parent] = append(childrenOf[parent], id)
	}
	for _, children := range childrenOf {
		sort.Strings(children)
	}

	var order []string
	reached := 0
	for queue := []string{world}; len(queue) > 0; queue = queue[1:] {
		id := queue[0]
		if e, ok := elements[id]; ok {
			reached++
			if e.isJoint && !e.mimic {
				order = append(order, id)
			}
		}
		queue = append(queue, childrenOf[id]...)
	}
	if reached != len(elements) {
		return nil, errors.Errorf("trajex/viamapi: model %q has links or joints that do not descend from its root", label)
	}
	return order, nil
}

// quaternion is w, x, y, z in the Hamilton convention, which is what trajex
// takes as a joint's origin.
type quaternion [4]float64

// pose is a rigid transform, a translation followed by a rotation, with the
// rotation kept as a unit quaternion.
type pose struct {
	xyz  [3]float64
	wxyz quaternion
}

var identityPose = pose{wxyz: quaternion{1, 0, 0, 0}}

func compose(a, b pose) pose {
	t := rotate(a.wxyz, b.xyz)
	return pose{
		xyz:  [3]float64{a.xyz[0] + t[0], a.xyz[1] + t[1], a.xyz[2] + t[2]},
		wxyz: multiply(a.wxyz, b.wxyz),
	}
}

func multiply(a, b quaternion) quaternion {
	return quaternion{
		a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
		a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
		a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
		a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
	}
}

// rotate applies the unit quaternion q to v as q v q*.
func rotate(q quaternion, v [3]float64) [3]float64 {
	conjugate := quaternion{q[0], -q[1], -q[2], -q[3]}
	p := multiply(multiply(q, quaternion{0, v[0], v[1], v[2]}), conjugate)
	return [3]float64{p[1], p[2], p[3]}
}

func vector(v *commonpb.Vector3) [3]float64 {
	return [3]float64{v.GetX(), v.GetY(), v.GetZ()}
}

func radians(degrees float64) float64 {
	return degrees * math.Pi / 180
}

// poseFromLinkPose reads a link's pose, whose orientation may be given any of
// the four ways the message allows. Angles on the wire are in degrees.
func poseFromLinkPose(lp *commonpb.LinkPose) (pose, error) {
	p := identityPose
	if lp == nil {
		return p, nil
	}
	p.xyz = vector(lp.GetTranslation())
	switch o := lp.GetOrientation().(type) {
	case nil:
	case *commonpb.LinkPose_Quaternion:
		q := o.Quaternion
		p.wxyz = unitQuaternion(quaternion{q.GetW(), q.GetX(), q.GetY(), q.GetZ()})
	case *commonpb.LinkPose_OrientationVector:
		ov := o.OrientationVector
		if ov.GetOX() == 0 && ov.GetOY() == 0 && ov.GetOZ() == 0 {
			// RDK refuses a link whose orientation vector has no direction, though it lets a part's pose have one
			return p, errors.New("orientation vector is zero")
		}
		p.wxyz = orientationVectorQuaternion(ov.GetOX(), ov.GetOY(), ov.GetOZ(), ov.GetTheta())
	case *commonpb.LinkPose_EulerAngles:
		e := o.EulerAngles
		p.wxyz = eulerQuaternion(radians(e.GetRoll()), radians(e.GetPitch()), radians(e.GetYaw()))
	case *commonpb.LinkPose_AxisAngles:
		aa := o.AxisAngles
		q, err := axisAngleQuaternion([3]float64{aa.GetRx(), aa.GetRy(), aa.GetRz()}, radians(aa.GetTheta()))
		if err != nil {
			return p, err
		}
		p.wxyz = q
	default:
		return p, errors.Errorf("orientation %T is not one this package reads", o)
	}
	return p, nil
}

// poseFromProto reads the pose a frame system gives a part, which is always a
// translation in millimetres and an orientation vector with theta in degrees.
func poseFromProto(pb *commonpb.Pose) pose {
	if pb == nil {
		return identityPose
	}
	return pose{
		xyz:  [3]float64{pb.GetX(), pb.GetY(), pb.GetZ()},
		wxyz: orientationVectorQuaternion(pb.GetOX(), pb.GetOY(), pb.GetOZ(), pb.GetTheta()),
	}
}

// unitQuaternion scales q to unit length. RDK normalizes a quaternion as it
// reads one from the message and reads a zero quaternion as no rotation, so
// we do the same for the two to agree.
func unitQuaternion(q quaternion) quaternion {
	norm := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
	if norm == 0 {
		return identityPose.wxyz
	}
	return quaternion{q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm}
}

// orientationVectorQuaternion is RDK's OrientationVector.Quaternion carried
// over, so the message reads the same here as there. The vector is where the
// frame's z axis points and theta, in degrees, turns about it. Near either
// pole RDK measures theta as if the vector had no longitude, so we fold the
// longitude back in to keep that meaning.
func orientationVectorQuaternion(ox, oy, oz, thetaDegrees float64) quaternion {
	theta := radians(thetaDegrees)
	norm := math.Sqrt(ox*ox + oy*oy + oz*oz)
	if norm == 0 {
		// RDK reads an unset vector in a part's pose as pointing along z
		ox, oy, oz = 0, 0, 1
	} else {
		ox, oy, oz = ox/norm, oy/norm, oz/norm
	}
	lat := math.Acos(oz)
	lon := math.Atan2(oy, ox)
	if 1-math.Abs(oz) <= orientationVectorPoleRadius {
		if oz > 0 {
			theta -= lon
		} else {
			theta += lon
		}
	}
	return zyzQuaternion(lon, lat, theta)
}

// zyzQuaternion is a rotation of a about z, then b about the new y, then c
// about the newest z, written out in the order of operations RDK's math
// library uses so the two agree.
func zyzQuaternion(a, b, c float64) quaternion {
	s0, c0 := math.Sincos(a / 2)
	s1, c1 := math.Sincos(b / 2)
	s2, c2 := math.Sincos(c / 2)
	return quaternion{
		c0*c1*c2 - s0*c1*s2,
		c0*s1*s2 - s0*s1*c2,
		c0*s1*c2 + s0*s1*s2,
		s0*c1*c2 + c0*c1*s2,
	}
}

// eulerQuaternion is RDK's EulerAngles.Quaternion: roll about x, pitch about
// y, yaw about z, all in radians.
func eulerQuaternion(roll, pitch, yaw float64) quaternion {
	cy := math.Cos(yaw * 0.5)
	sy := math.Sin(yaw * 0.5)
	cp := math.Cos(pitch * 0.5)
	sp := math.Sin(pitch * 0.5)
	cr := math.Cos(roll * 0.5)
	sr := math.Sin(roll * 0.5)
	return quaternion{
		cr*cp*cy + sr*sp*sy,
		sr*cp*cy - cr*sp*sy,
		cr*sp*cy + sr*cp*sy,
		cr*cp*sy - sr*sp*cy,
	}
}

// axisAngleQuaternion turns theta radians about axis, which need not be of
// unit length. It returns an error for a zero axis.
func axisAngleQuaternion(axis [3]float64, theta float64) (quaternion, error) {
	norm := math.Sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
	if norm == 0 {
		return quaternion{}, errors.New("axis angles have a zero axis")
	}
	s := math.Sin(theta / 2)
	return quaternion{math.Cos(theta / 2), axis[0] / norm * s, axis[1] / norm * s, axis[2] / norm * s}, nil
}
