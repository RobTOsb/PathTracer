#pragma once

#include <DirectXMath.h>
#include <DirectXPackedVector.h>

using namespace DirectX;

constexpr float Pi = 3.14159265358979323846f;

using uint = uint32_t;
using uint2 = XMUINT2;
using uint3 = XMUINT3;
using uint4 = XMUINT4;

using int2 = XMINT2;
using int3 = XMINT3;
using int4 = XMINT4;

using int16 = int16_t;
using uint64 = uint64_t;

// Type aliases for easier migration
using float2 = XMFLOAT2;
using float3 = XMFLOAT3;
using float4 = XMFLOAT4;
using float3x3 = XMFLOAT3X3;
using float3x4 = XMFLOAT3X4;
using float4x4 = XMFLOAT4X4;
using Quaternion = XMFLOAT4; // Quaternion stored as (x, y, z, w)

using SIMDVector = XMVECTOR;
using SIMDMatrix = XMMATRIX;

// Helper functions for vector operations
inline SIMDVector LoadFloat3(const float3 &f3)
{
	return XMLoadFloat3(&f3);
}

inline SIMDVector LoadFloat4(const float4 &f4)
{
	return XMLoadFloat4(&f4);
}

inline SIMDMatrix LoadFloat4x4(const float4x4 &f4x4)
{
	return XMLoadFloat4x4(&f4x4);
}

inline void StoreFloat3(float3 &f3, SIMDVector vec)
{
	XMStoreFloat3(&f3, vec);
}

inline void StoreFloat4(float4 &f4, SIMDVector vec)
{
	XMStoreFloat4(&f4, vec);
}

inline void StoreFloat4x4(float4x4 &f4x4, SIMDMatrix mat)
{
	XMStoreFloat4x4(&f4x4, mat);
}

// Vector operations
inline float2 operator-(const float2 &a, const float2 &b)
{
	return float2(a.x - b.x, a.y - b.y);
}

inline float3 operator+(const float3 &a, const float3 &b)
{
	float3 result;
	XMStoreFloat3(&result, XMVectorAdd(XMLoadFloat3(&a), XMLoadFloat3(&b)));
	return result;
}

inline float3 operator*(const float3 &a, float scalar)
{
	float3 result;
	XMStoreFloat3(&result, XMVectorScale(XMLoadFloat3(&a), scalar));
	return result;
}

inline float3 operator*(const float3 &a, const float3 &b)
{
	float3 result;
	XMStoreFloat3(&result, XMVectorMultiply(XMLoadFloat3(&a), XMLoadFloat3(&b)));
	return result;
}

inline float3 operator-(const float3 &a, const float3 &b)
{
	float3 result;
	XMStoreFloat3(&result, XMVectorSubtract(XMLoadFloat3(&a), XMLoadFloat3(&b)));
	return result;
}

inline float4 operator+(const float4 &a, const float4 &b)
{
	float4 result;
	XMStoreFloat4(&result, XMVectorAdd(XMLoadFloat4(&a), XMLoadFloat4(&b)));
	return result;
}

inline float4 operator-(const float4 &a, const float4 &b)
{
	float4 result;
	XMStoreFloat4(&result, XMVectorSubtract(XMLoadFloat4(&a), XMLoadFloat4(&b)));
	return result;
}

inline float4 operator*(const float4 &a, float scalar)
{
	float4 result;
	XMStoreFloat4(&result, XMVectorScale(XMLoadFloat4(&a), scalar));
	return result;
}

inline float4 operator*(const float4 &a, const float4 &b)
{
	float4 result;
	XMStoreFloat4(&result, XMVectorMultiply(XMLoadFloat4(&a), XMLoadFloat4(&b)));
	return result;
}


// Cross product
inline float3 Cross(const float3 &a, const float3 &b)
{
	float3 result;
	XMStoreFloat3(&result, XMVector3Cross(XMLoadFloat3(&a), XMLoadFloat3(&b)));
	return result;
}

inline float4 Cross(const float4 &a, const float4 &b)
{
	float4 result;
	XMStoreFloat4(&result, XMVector3Cross(XMLoadFloat4(&a), XMLoadFloat4(&b)));
	return result;
}

// Dot product
inline float Dot(const float3 &a, const float3 &b)
{
	XMVECTOR dotVec = XMVector3Dot(XMLoadFloat3(&a), XMLoadFloat3(&b));
	return XMVectorGetX(dotVec);
}

inline float Dot(const float4 &a, const float4 &b)
{
	XMVECTOR dotVec = XMVector3Dot(XMLoadFloat4(&a), XMLoadFloat4(&b));
	return XMVectorGetX(dotVec);
}

// Quaternion operations
inline Quaternion CreateQuaternionFromAxisAngle(const float3 &axis, float angleRadians)
{
	Quaternion result;
	XMStoreFloat4(&result, XMQuaternionRotationAxis(XMLoadFloat3(&axis), angleRadians));
	return result;
}

inline Quaternion CreateQuaternionFromAxisAngle(const float4 &axis, float angleRadians)
{
	Quaternion result;
	XMStoreFloat4(&result, XMQuaternionRotationAxis(XMLoadFloat4(&axis), angleRadians));
	return result;
}

inline float3 RotateVectorByQuaternion(const float3 &vec, const Quaternion &quat)
{
	float3 result;
	XMStoreFloat3(&result, XMVector3Rotate(XMLoadFloat3(&vec), XMLoadFloat4(&quat)));
	return result;
}

inline void NormaliseQuaternion(Quaternion &quat)
{
	XMStoreFloat4(&quat, XMQuaternionNormalize(XMLoadFloat4(&quat)));
}

inline void NormaliseFloat3(float3 &vec)
{
	XMStoreFloat3(&vec, XMVector3Normalize(XMLoadFloat3(&vec)));
}

inline void NormaliseFloat4(float4 &vec)
{
	XMStoreFloat4(&vec, XMVector4Normalize(XMLoadFloat4(&vec)));
}

inline Quaternion MultiplyQuaternions(const Quaternion &q1, const Quaternion &q2)
{
	Quaternion result;
	XMStoreFloat4(&result, XMQuaternionMultiply(XMLoadFloat4(&q1), XMLoadFloat4(&q2)));
	return result;
}

// Matrix operations
inline float4x4 CreateIdentityMatrix()
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixIdentity());
	return result;
}

inline float4x4 CreateTranslationMatrix(float tx, float ty, float tz)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixTranslation(tx, ty, tz));
	return result;
}

inline float4x4 CreateTranslationMatrix(const float4 &translationVec)
{
	return CreateTranslationMatrix(translationVec.x, translationVec.y, translationVec.z);
}

inline float4x4 CreateRotationMatrixX(float angleRadians)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixRotationX(angleRadians));
	return result;
}

inline float4x4 CreateRotationMatrixY(float angleRadians)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixRotationY(angleRadians));
	return result;
}

inline float4x4 CreateRotationMatrixZ(float angleRadians)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixRotationZ(angleRadians));
	return result;
}

inline float4x4 CreateLookAtMatrixRH(const float3 &eye, const float3 &target, const float3 &up)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixLookAtRH(XMLoadFloat3(&eye), XMLoadFloat3(&target), XMLoadFloat3(&up)));
	return result;
}

inline float4x4 CreatePerspectiveMatrixFovRH(float fovY, float aspect, float nearZ, float farZ)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixPerspectiveFovRH(fovY, aspect, nearZ, farZ));
	return result;
}

inline XMMATRIX ReversedZInfinitePerspectiveRH(float fovY, float aspect, float zNear)
{
    float h = 1.0f / tanf(fovY * 0.5f);
    float w = h / aspect;

    XMMATRIX m = {};
    m.r[0] = XMVectorSet(w,    0.0f,  0.0f,   0.0f);
    m.r[1] = XMVectorSet(0.0f, h,     0.0f,   0.0f);
    m.r[2] = XMVectorSet(0.0f, 0.0f,  0.0f,  -1.0f);  // W copy, negative for RH
    m.r[3] = XMVectorSet(0.0f, 0.0f,  zNear,  0.0f);  // zNear term, no negation needed

    return m;
}
inline float4x4 CreatePerspectiveMatrixFovReverseZInfiniteRH(float fovY, float aspect, float nearZ)
{
	float4x4 result;
	XMStoreFloat4x4(&result, ReversedZInfinitePerspectiveRH(fovY, aspect, nearZ));
	return result;
}

inline float4x4 operator*(const float4x4 &a, const float4x4 &b)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixMultiply(XMLoadFloat4x4(&a), XMLoadFloat4x4(&b)));
	return result;
}

inline float4x4 operator+(const float4x4 &a, const float4x4 &b)
{
	float4x4 result;
	XMMATRIX m1 = XMLoadFloat4x4(&a);
	XMMATRIX m2 = XMLoadFloat4x4(&b);
	XMStoreFloat4x4(&result, m1 + m2);
	return result;
}

inline float4 operator*(const float4 &vec, const float4x4 &mat)
{
	float4 result;
	XMStoreFloat4(&result, XMVector4Transform(XMLoadFloat4(&vec), XMLoadFloat4x4(&mat)));
	return result;
}

inline float4x4 TransposeMatrix(const float4x4 &mat)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixTranspose(XMLoadFloat4x4(&mat)));
	return result;
}

// Utility functions
inline float Clamp(float value, float min, float max)
{
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

inline constexpr float DegreesToRadians(float degrees)
{
	return degrees * Pi / 180.0f;
}

inline float4x4 InvertMatrix(const float4x4 &mat)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixInverse(nullptr, XMLoadFloat4x4(&mat)));
	return result;
}

inline float4x4 CreateScalingMatrix(float sx, float sy, float sz)
{
	float4x4 result;
	XMStoreFloat4x4(&result, XMMatrixScaling(sx, sy, sz));
	return result;
}

inline float4x4 CreateScalingMatrix(const float3 &scaleVec)
{
	return CreateScalingMatrix(scaleVec.x, scaleVec.y, scaleVec.z);
}

// Decomposes a TRS matrix into its translation/rotation/scale components.
// Returns false (DirectXMath failure, e.g. a degenerate/non-invertible matrix) without
// modifying the outputs.
inline bool DecomposeTransform(const float4x4 &mat, float3 &outTranslation, Quaternion &outRotationQuat, float3 &outScale)
{
	XMVECTOR s, r, t;
	if (!XMMatrixDecompose(&s, &r, &t, LoadFloat4x4(mat)))
		return false;

	StoreFloat3(outScale, s);
	StoreFloat4(outRotationQuat, r);
	StoreFloat3(outTranslation, t);
	return true;
}

// Converts a quaternion to intrinsic pitch(X)/yaw(Y)/roll(Z) Euler angles in radians,
// matching the axis order expected by XMQuaternionRotationRollPitchYaw(x, y, z).
inline float3 QuaternionToEulerRadians(const Quaternion &q)
{
	const float sinPitch = 2.0f * (q.w * q.x - q.y * q.z);
	float pitch;
	if (fabsf(sinPitch) >= 1.0f)
		pitch = copysignf(Pi * 0.5f, sinPitch);
	else
		pitch = asinf(sinPitch);

	const float yaw = atan2f(2.0f * (q.w * q.y + q.x * q.z), 1.0f - 2.0f * (q.x * q.x + q.y * q.y));
	const float roll = atan2f(2.0f * (q.w * q.z + q.x * q.y), 1.0f - 2.0f * (q.x * q.x + q.z * q.z));

	return float3{ pitch, yaw, roll };
}
