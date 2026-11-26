#pragma once

#include "AnimationData.h"

namespace Animation
{
	class AnimationSystem
	{
	public:
		AnimationSystem() = default;
		explicit AnimationSystem(const AnimationData &data)
		{
			Reset(data);
		}

		// Takes new tables (a scene load), which must outlive this. The pose is empty until the next Evaluate.
		void Reset(const AnimationData &data);

		// Nothing to pose: no channels, joints or skinned meshes.
		bool IsEmpty() const
		{
			return data_.nodes.empty();
		}

		bool HasAnimation() const
		{
			return !data_.channels.empty();
		}

		bool HasSkins() const
		{
			return !data_.skinnedPrimitives.empty();
		}

		// The time of the last key of any channel, which the animations loop over.
		float GetDuration() const
		{
			return duration_;
		}

		uint32_t GetSkinCount() const
		{
			return static_cast<uint32_t>(data_.skins.size());
		}

		uint32_t GetJointCount() const
		{
			return static_cast<uint32_t>(data_.joints.size());
		}

		uint32_t GetSkinnedVertexCount() const
		{
			return static_cast<uint32_t>(data_.bindVertices.size());
		}

		// Poses every node at this many seconds into the animations and skins the meshes to match.
		const Pose &Evaluate(float seconds);

		// As the last Evaluate left it.
		const Pose &GetPose() const
		{
			return pose_;
		}

		// CPU time of the last Evaluate, and the part of it spent skinning.
		double GetEvaluateMs() const
		{
			return evaluateMs_;
		}

		double GetSkinningMs() const
		{
			return skinningMs_;
		}

	private:
		// A node's local transform in parts, the rest pose with what the channels animate replaced.
		struct LocalPose
		{
			float translation[3];
			float rotation[4];
			float scale[3];
			bool animated;
		};

		void PoseNodes(float time);
		void PlaceInstances();
		void SkinMeshes();

		AnimationData data_;
		float duration_ = 0.0f;
		// Per node, whether the instance it places is drawn skinned, so the skin places it instead.
		std::vector<uint8_t> placesSkinnedInstance_;

		Pose pose_;
		std::vector<LocalPose> localPoses_;
		std::vector<Matrix3x4> jointMatrices_; // per joint of every skin, from the bind pose to the anchor's space
		double evaluateMs_ = 0.0;
		double skinningMs_ = 0.0;
	};
} // namespace Animation
