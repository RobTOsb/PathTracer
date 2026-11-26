#include "AnimationSystem.h"

#include "Skinning.h"

#include "Instance.h" // GLTFScene's: Scene::ComposeTRS, Scene::ComposeAffine

#include <algorithm>
#include <chrono>
#include <cmath>

namespace Animation
{
	namespace
	{
		// Inverse of an affine float3x4 in the scene layout, which maps p to p A + t with A[r][c] = m[4 r + c] and
		// t = (m[3], m[7], m[11]).
		Matrix3x4 InvertAffine(const Matrix3x4 &m)
		{
			const float det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
							  m[2] * (m[4] * m[9] - m[5] * m[8]);
			if (det == 0.0f || !std::isfinite(det))
			{
				return kIdentity;
			}
			const float s = 1.0f / det;
			Matrix3x4 inv;
			inv[0] = (m[5] * m[10] - m[6] * m[9]) * s;
			inv[1] = (m[2] * m[9] - m[1] * m[10]) * s;
			inv[2] = (m[1] * m[6] - m[2] * m[5]) * s;
			inv[4] = (m[6] * m[8] - m[4] * m[10]) * s;
			inv[5] = (m[0] * m[10] - m[2] * m[8]) * s;
			inv[6] = (m[2] * m[4] - m[0] * m[6]) * s;
			inv[8] = (m[4] * m[9] - m[5] * m[8]) * s;
			inv[9] = (m[1] * m[8] - m[0] * m[9]) * s;
			inv[10] = (m[0] * m[5] - m[1] * m[4]) * s;
			// p = (p' - t) A^-1, so the translation is -t A^-1.
			for (int c = 0; c < 3; ++c)
			{
				inv[4 * c + 3] = -(m[3] * inv[c] + m[7] * inv[4 + c] + m[11] * inv[8 + c]);
			}
			return inv;
		}

		double MillisecondsSince(std::chrono::steady_clock::time_point start)
		{
			return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		}
	} // namespace

	void AnimationSystem::Reset(const AnimationData &data)
	{
		data_ = data;
		pose_ = {};
		localPoses_.assign(data_.nodes.size(), {});
		jointMatrices_.assign(data_.joints.size(), kIdentity);

		duration_ = 0.0f;
		for (const Channel &channel: data_.channels)
		{
			duration_ = std::max(duration_, data_.times[channel.firstTime + channel.keyCount - 1]);
		}

		std::vector<uint8_t> skinnedInstance;
		for (const SkinnedPrimitive &primitive: data_.skinnedPrimitives)
		{
			if (primitive.instance >= skinnedInstance.size())
			{
				skinnedInstance.resize(primitive.instance + 1, 0);
			}
			skinnedInstance[primitive.instance] = 1;
		}
		placesSkinnedInstance_.assign(data_.nodes.size(), 0);
		for (size_t i = 0; i < data_.nodes.size(); ++i)
		{
			const uint32_t instance = data_.nodes[i].instance;
			placesSkinnedInstance_[i] = instance < skinnedInstance.size() && skinnedInstance[instance];
		}
	}

	const Pose &AnimationSystem::Evaluate(float seconds)
	{
		const auto start = std::chrono::steady_clock::now();
		pose_.seconds = duration_ > 0.0f ? seconds - std::floor(seconds / duration_) * duration_ : 0.0f;
		PoseNodes(pose_.seconds);
		PlaceInstances();
		const auto skinStart = std::chrono::steady_clock::now();
		SkinMeshes();
		skinningMs_ = MillisecondsSince(skinStart);
		evaluateMs_ = MillisecondsSince(start);
		return pose_;
	}

	void AnimationSystem::PoseNodes(float time)
	{
		for (size_t i = 0; i < data_.nodes.size(); ++i)
		{
			const Node &node = data_.nodes[i];
			LocalPose &pose = localPoses_[i];
			std::copy(std::begin(node.translation), std::end(node.translation), pose.translation);
			std::copy(std::begin(node.rotation), std::end(node.rotation), pose.rotation);
			std::copy(std::begin(node.scale), std::end(node.scale), pose.scale);
			pose.animated = false;
		}
		for (const Channel &channel: data_.channels)
		{
			LocalPose &pose = localPoses_[channel.node];
			float *target = channel.path == Scene::AnimationPath::Translation ? pose.translation
							: channel.path == Scene::AnimationPath::Rotation  ? pose.rotation
																			  : pose.scale;
			float value[4];
			Scene::SampleAnimation(channel.path,
								   channel.interpolation,
								   data_.times.data() + channel.firstTime,
								   data_.values.data() + channel.firstValue,
								   channel.keyCount,
								   time,
								   value);
			std::copy(value, value + Scene::AnimationPathComponents(channel.path), target);
			pose.animated = true;
		}

		// Parents come first, so their world matrices are ready for their children.
		pose_.nodeWorlds.resize(data_.nodes.size());
		for (size_t i = 0; i < data_.nodes.size(); ++i)
		{
			const Node &node = data_.nodes[i];
			const LocalPose &pose = localPoses_[i];
			float composed[12];
			if (pose.animated)
			{
				Scene::ComposeTRS(pose.translation, pose.rotation, pose.scale, composed);
			}
			const float *local = pose.animated ? composed : node.localMatrix;
			const float *parent = node.parent == kNone ? node.parentMatrix : pose_.nodeWorlds[node.parent].data();
			Scene::ComposeAffine(local, parent, pose_.nodeWorlds[i].data());
		}
	}

	void AnimationSystem::PlaceInstances()
	{
		pose_.instances.clear();
		for (size_t i = 0; i < data_.nodes.size(); ++i)
		{
			const uint32_t instance = data_.nodes[i].instance;
			if (instance != kNone && !placesSkinnedInstance_[i])
			{
				pose_.instances.push_back({ instance, pose_.nodeWorlds[i] });
			}
		}
		// A skinned mesh ignores its node's transform: its vertices are in the anchor's space, so the anchor places it.
		uint32_t placed = kNone;
		for (const SkinnedPrimitive &primitive: data_.skinnedPrimitives)
		{
			if (primitive.instance != placed)
			{
				pose_.instances.push_back({ primitive.instance, pose_.nodeWorlds[data_.skins[primitive.skin].anchor] });
				placed = primitive.instance;
			}
		}
	}

	void AnimationSystem::SkinMeshes()
	{
		if (data_.skinnedPrimitives.empty())
		{
			pose_.skinnedVertices.clear();
			return;
		}

		// v_anchor = v_bind * inverseBind * jointWorld * anchorWorld^-1, the three composed once per joint.
		for (const Skin &skin: data_.skins)
		{
			const Matrix3x4 toAnchor = InvertAffine(pose_.nodeWorlds[skin.anchor]);
			for (uint32_t j = skin.firstJoint; j < skin.firstJoint + skin.jointCount; ++j)
			{
				const Joint &joint = data_.joints[j];
				float toWorld[12];
				Scene::ComposeAffine(joint.inverseBind, pose_.nodeWorlds[joint.node].data(), toWorld);
				Scene::ComposeAffine(toWorld, toAnchor.data(), jointMatrices_[j].data());
			}
		}

		pose_.skinnedVertices.resize(data_.bindVertices.size());
		for (const SkinnedPrimitive &primitive: data_.skinnedPrimitives)
		{
			const Skin &skin = data_.skins[primitive.skin];
			SkinVertices(std::span<const Matrix3x4>(jointMatrices_).subspan(skin.firstJoint, skin.jointCount),
						 data_.influences.subspan(primitive.firstVertex, primitive.vertexCount),
						 data_.bindVertices.subspan(primitive.firstVertex, primitive.vertexCount),
						 std::span<SkinVertex>(pose_.skinnedVertices).subspan(primitive.firstVertex, primitive.vertexCount));
		}
	}
} // namespace Animation
