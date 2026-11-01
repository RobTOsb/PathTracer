#include "Camera.h"

void Camera::RotateCamera(float yawDelta, float pitchDelta, float sensitivity)
{
	// Convert input to radians and apply sensitivity
	float pitchRadians = DegreesToRadians(pitchDelta * sensitivity);
	float yawRadians = DegreesToRadians(yawDelta * sensitivity);

	// Accumulate yaw and pitch angles
	yaw += yawRadians;
	pitch += pitchRadians;

	// Clamp pitch to prevent going over top or under bottom
	pitch = Clamp(pitch, negativeRotationClamp, rotationClamp);

	UpdateOrientation();
}

void Camera::SetOrientation(float newYaw, float newPitch)
{
	yaw = newYaw;
	pitch = Clamp(newPitch, negativeRotationClamp, rotationClamp);
	UpdateOrientation();
}

void Camera::SetView(const float3 &newPosition, const float3 &newFront, const float3 &newUp)
{
	position = newPosition;
	front = newFront;
	NormaliseFloat3(front);
	right = Cross(front, newUp);
	NormaliseFloat3(right);
	up = Cross(right, front);
	// The inverse of UpdateOrientation's front from yaw and pitch.
	yaw = atan2f(front.x, front.z);
	pitch = Clamp(asinf(Clamp(front.y, -1.0f, 1.0f)), negativeRotationClamp, rotationClamp);
	UpdateViewMatrix();
}

void Camera::UpdateOrientation()
{
	float3 worldUp = { 0.0f, 1.0f, 0.0f };
	
	// Calculate front vector from yaw and pitch
	front.x = cosf(pitch) * sinf(yaw);
	front.y = sinf(pitch);
	front.z = cosf(pitch) * cosf(yaw);
	NormaliseFloat3(front);

	// Calculate right and up vectors to maintain orthonormal basis
	right = Cross(front, worldUp);
	NormaliseFloat3(right);
	up = Cross(right, front);
	NormaliseFloat3(up);

	// In orbital mode, update position to maintain distance from target
	if (orbitalMode)
	{
		position = target - front * distanceFromTarget;
	}

	UpdateViewMatrix();
}