#pragma once


#include "Animation.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace Animation
{
	using Matrix3x4 = std::array<float, 12>;

	inline constexpr Matrix3x4 kIdentity = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };

	// No entry, for the indices below.
	inline constexpr uint32_t kNone = UINT32_MAX;

	// The skin influences a vertex has at most.
	inline constexpr uint32_t kMaxInfluences = 4;

	// A node the animation system poses: one a channel moves, a parent of one, a skin's joint or its anchor, or a
	// skinned mesh's node. The table lists every node after its parent.
	struct Node
	{
		uint32_t parent; // into the node table, or kNone for a parent that is never posed (its world is parentMatrix)
		uint32_t instance; // the scene instance the node places, or kNone (scene data, carried through to the pose)
		// The local transform at rest, in parts for the channels to replace, and whole for nodes without any.
		float translation[3];
		float rotation[4]; // quaternion xyzw
		float scale[3];
		float localMatrix[12];
		float parentMatrix[12]; // the world matrix of a parent that is never posed, identity for a root
	};

	// A glTF animation channel, its keys in AnimationData::times and ::values (see Scene::SampleAnimation).
	struct Channel
	{
		uint32_t node; // into the node table
		Scene::AnimationPath path;
		Scene::AnimationInterpolation interpolation;
		uint32_t keyCount;
		uint32_t firstTime;
		uint32_t firstValue;
	};

	// A glTF skin. Its meshes' vertices come out in the space of its anchor node, which places their scene instance:
	// moving the whole skeleton then moves the instance, and only the bending is in the vertices.
	struct Skin
	{
		uint32_t firstJoint; // into the joint table
		uint32_t jointCount;
		uint32_t anchor; // into the node table: the skin's skeleton root, or else its first joint
	};

	struct Joint
	{
		uint32_t node; // into the node table
		float inverseBind[12]; // from the mesh's bind pose into the joint's space
	};

	// A run of vertices a skin deforms: one primitive of a skinned mesh.
	struct SkinnedPrimitive
	{
		uint32_t skin; // into the skin table
		uint32_t instance; // the scene instance it is drawn with, which the pose places at the skin's anchor
		uint32_t primitive; // the scene's own name for the vertices (the GPU scene's primitive desc), carried through
		uint32_t firstVertex; // into the influence and bind vertex tables, and the pose's skinned vertices
		uint32_t vertexCount;
	};

	struct VertexInfluence
	{
		uint16_t joints[kMaxInfluences]; // into the skin's joints
		float weights[kMaxInfluences]; // sum to 1
	};

	// What skinning moves of a vertex.
	struct SkinVertex
	{
		float position[3];
		float normal[3];
		float tangent[4]; // xyz, w the bitangent sign
	};

	// A view of a scene's tables, which must outlive the animation system made from it.
	struct AnimationData
	{
		std::span<const Node> nodes;
		std::span<const Channel> channels;
		std::span<const float> times;
		std::span<const float> values;
		std::span<const Skin> skins;
		std::span<const Joint> joints;
		std::span<const SkinnedPrimitive> skinnedPrimitives;
		std::span<const VertexInfluence> influences; // per skinned vertex
		std::span<const SkinVertex> bindVertices; // per skinned vertex, in the mesh's space
	};

	// Where a pose puts a scene instance.
	struct InstancePlacement
	{
		uint32_t instance;
		Matrix3x4 world;
	};

	// The animation system's output at one time.
	struct Pose
	{
		float seconds = 0.0f; // into the animations, wrapped to their duration
		std::vector<Matrix3x4> nodeWorlds; // per node of the table
		// Every scene instance a node places, or a skin (at its anchor), in node then skinned primitive order.
		std::vector<InstancePlacement> instances;
		// Per skinned vertex, in its skin's anchor space.
		std::vector<SkinVertex> skinnedVertices;
	};
} // namespace Animation
