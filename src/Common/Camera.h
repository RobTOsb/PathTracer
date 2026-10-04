#pragma once

#include "Maths.h"
#include <limits>

class Camera
{
public:
	Camera(float fovY = 90.0f,
		   float aspectRatio = 16.0f / 9.0f,
		   float nearZ = 0.0001f,
		   float farZ = 1000.0f,
		   bool isOrbital = false,
		   bool isReversedZ = true)
	{
		position = { 0.0f, 0.0f, 0.1f };
		front = { 0.0f, 0.0f, -1.0f };
		up = { 0.0f, 1.0f, 0.0f };
		right = { 1.0f, 0.0f, 0.0f };
		target = { 0.0f, 0.0f, 0.0f }; // Centre of the world for orbital mode

		// Initialize yaw and pitch to match initial front vector (0, 0, -1)
		yaw = Pi; // Looking towards -Z
		pitch = 0.0f; // Level camera

		viewMatrix = CreateLookAtMatrixRH(position, position + front, up);

		this->fovY = DegreesToRadians(fovY);

		if (isReversedZ)
		{
			projectionMatrix = CreatePerspectiveMatrixFovReverseZInfiniteRH(this->fovY, aspectRatio, nearZ);
			this->nearZ = nearZ;
			this->farZ = std::numeric_limits<float>::infinity();
		}
		else
		{
			projectionMatrix = CreatePerspectiveMatrixFovRH(this->fovY, aspectRatio, nearZ, farZ);
			this->nearZ = nearZ;
			this->farZ = farZ;
		}


		prevViewMatrix = viewMatrix;
		prevProjectionMatrix = projectionMatrix;

		orbitalMode = isOrbital;

		ev100 = log2((aperture * aperture) / shutterSpeed * 100.0f /
					 ISO); // Calculate EV100 based on aperture, shutter speed, and ISO
		exposureMultiplier =
				1.0f / 1.2f * pow(2.0f, ev100); // Calculate exposure multiplier for tonemapping based on EV100
	}

	float3 GetPosition() const
	{
		return position;
	}

	float3 GetFront() const
	{
		return front;
	}

	float3 GetUp() const
	{
		return up;
	}

	float3 GetRight() const
	{
		return right;
	}

	float4x4 GetViewMatrix() const
	{
		return viewMatrix;
	}

	float4x4 GetProjectionMatrix() const
	{
		return projectionMatrix;
	}

	float4x4 GetPreviousViewMatrix() const
	{
		return prevViewMatrix;
	}

	float4x4 GetPreviousProjectionMatrix() const
	{
		return prevProjectionMatrix;
	}

	float GetNearZ() const
	{
		return nearZ;
	}

	float GetFarZ() const
	{
		return farZ;
	}

	float GetFovY() const // radians
	{
		return fovY;
	}


	bool IsOrbital() const
	{
		return orbitalMode;
	}

	void SetOrbitalMode(bool isOrbital)
	{
		orbitalMode = isOrbital;
	}

	void SetDistanceFromTarget(float distance)
	{
		distanceFromTarget = distance;
		position = target - front * distanceFromTarget; // Update position based on new distance
		UpdateViewMatrix();
	}

	void RecalculateExposureValues()
	{
		ev100 = log2((aperture * aperture) / shutterSpeed * 100.0f /
					 ISO); // Recalculate EV100 based on current aperture, shutter speed, and ISO
		exposureMultiplier =
				1.0f / 1.2f * pow(2.0f, ev100); // Recalculate exposure multiplier for tonemapping based on new EV100
	}

	void SetAperture(float newAperture)
	{
		aperture = newAperture;
	}

	void SetShutterSpeed(float newShutterSpeed)
	{
		shutterSpeed = newShutterSpeed;
	}

	void SetISO(float newISO)
	{
		ISO = newISO;
	}

	float GetExposureMultiplier() const
	{
		return exposureMultiplier;
	}

	float GetEV100() const
	{
		return ev100;
	}

	float GetAperture() const
	{
		return aperture;
	}

	float GetShutterSpeed() const
	{
		return shutterSpeed;
	}

	float GetISO() const
	{
		return ISO;
	}

	void Zoom(float distance)
	{
		if (orbitalMode)
		{
			distanceFromTarget += distance;
			position = target - front * distanceFromTarget; // Move camera closer/further from target
		}

		UpdateViewMatrix();
	}

	void SetTarget(const float3 &newTarget)
	{
		target = newTarget;
	}

	void SetPosition(const float3 &newPosition)
	{
		position = newPosition;
		UpdateViewMatrix();
	}

	void MoveForward(float distance)
	{
		position = position + front * distance;
		UpdateViewMatrix();
	}

	void StrafeRight(float distance)
	{
		position = position + right * distance;
		UpdateViewMatrix();
	}

	void StrafeLeft(float distance)
	{
		position = position - right * distance;
		UpdateViewMatrix();
	}

	void MoveBackward(float distance)
	{
		position = position - front * distance;
		UpdateViewMatrix();
	}

	void UpdatePreviousMatrices()
	{
		prevViewMatrix = viewMatrix;
		prevProjectionMatrix = projectionMatrix;
	}

	void UpdateViewMatrix()
	{
		if (orbitalMode)
		{
			viewMatrix = CreateLookAtMatrixRH(position, target, up);
		}
		else
		{
			// In free mode, we update the view matrix based on the camera's position and orientation
			viewMatrix = CreateLookAtMatrixRH(position, position + front, up);
		}
	}

	void UpdateProjectionMatrix(float fovY, float aspectRatio, float nearZ, float farZ)
	{
		this->fovY = DegreesToRadians(fovY);
		projectionMatrix = CreatePerspectiveMatrixFovRH(this->fovY, aspectRatio, nearZ, farZ);
		this->nearZ = nearZ;
		this->farZ = farZ;
	}

	void RotateCamera(float yaw, float pitch, float sensitivity = 0.05f);

private:
	float3 position;
	float3 front;
	float3 up;
	float3 right;
	float3 target; // Used in orbital mode to store the point the camera is orbiting around
	float nearZ = 0.0f; // Padding to align to 16 bytes
	float farZ = 0.0f; // Padding to align to 16 bytes
	float fovY = 0.0f; // radians
	float aperture = 16.0f; //
	float shutterSpeed = 1.0f / 100.0f;
	float ISO = 100.0f;
	float ev100 = 0.0f; // Exposure value at ISO 100, calculated from aperture and shutter speed
	float exposureMultiplier = 0.0f; // Final value for tonemapping

	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	float4x4 prevProjectionMatrix;
	float4x4 prevViewMatrix;

	const float rotationClamp = Pi / 2.0f - 0.01f; // Prevent gimbal lock
	const float negativeRotationClamp = -rotationClamp;
	float distanceFromTarget = 0.1f; // Used in orbital mode to maintain a fixed distance from the target point
	float yaw = 0.0f; // Accumulated yaw angle (rotation around world Y axis)
	float pitch = 0.0f; // Accumulated pitch angle (rotation around right axis)

	bool orbitalMode; // If true, the camera will orbit around a target point instead of rotating in place
};
