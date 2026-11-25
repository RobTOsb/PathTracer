#include "LensFlarePass.h"

#include "BloomPass.h"
#include "Camera.h"
#include "FrameGraph.h"
#include "Scene/LightTreeBuilder.h"
#include "Scene/RenderGPUScene.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace
{
	constexpr double kPi = 3.14159265358979323846;

	// A lens prescription, front to back: radius of curvature (0 = flat), distance to the next surface, refractive
	// index behind the surface at the d line (0 marks the aperture stop, a plane that only clips), clear diameter
	// (mm), and that glass's Abbe number (0 for air). The last gap is not used: the sensor sits at the paraxial focus.
	struct LensRow
	{
		float radius;
		float thickness;
		float ior;
		float diameter;
		float abbe;
	};

	// Double-Gauss 50 mm f/2 (US patent 2,673,491, Tronnier; Smith, "Modern Lens Design" p. 312, scaled to 50 mm, as
	// in pbrt's dgauss.50mm).
	constexpr LensRow kDoubleGauss[] = {
		{ 29.475f, 3.76f, 1.67f, 25.2f, 47.2f },   { 84.83f, 0.12f, 1.0f, 25.2f, 0.0f },
		{ 19.275f, 4.025f, 1.67f, 23.0f, 47.2f },  { 40.77f, 3.275f, 1.699f, 23.0f, 30.1f },
		{ 12.75f, 5.705f, 1.0f, 18.0f, 0.0f },	   { 0.0f, 4.5f, 0.0f, 17.1f, 0.0f },
		{ -14.495f, 1.18f, 1.603f, 17.0f, 38.0f }, { 40.77f, 6.065f, 1.658f, 20.0f, 50.9f },
		{ -20.385f, 0.19f, 1.0f, 20.0f, 0.0f },	   { 437.065f, 3.22f, 1.717f, 20.0f, 47.9f },
		{ -39.73f, 5.0f, 1.0f, 20.0f, 0.0f },
	};

	// Angenieux double Gauss 50 mm (LensFlareFramework's double-gauss-angenieux; heights there are diameters).
	constexpr LensRow kAngenieux[] = {
		{ 82.06f, 5.495f, 1.673f, 27.0f, 32.2f },	{ 279.64f, 0.115f, 1.0f, 27.0f, 0.0f },
		{ 50.06f, 5.725f, 1.670f, 25.5f, 47.1f },	{ 106.77f, 0.115f, 1.0f, 25.5f, 0.0f },
		{ 29.02f, 11.475f, 1.691f, 20.5f, 54.7f },	{ 1275.5f, 1.29f, 1.673f, 20.5f, 32.2f },
		{ 16.195f, 7.83f, 1.0f, 13.5f, 0.0f },		{ 0.0f, 7.5f, 0.0f, 12.75f, 0.0f },
		{ -20.21f, 1.37f, 1.699f, 12.5f, 30.1f },	{ 96.49f, 13.96f, 1.620f, 18.0f, 60.3f },
		{ -27.765f, 0.115f, 1.0f, 18.0f, 0.0f },	{ 96.49f, 3.99f, 1.691f, 17.5f, 54.7f },
		{ -112.64f, 0.115f, 1.0f, 17.5f, 0.0f },	{ 87.55f, 4.24f, 1.691f, 17.5f, 54.7f },
		{ -101.77f, 27.871f, 1.0f, 17.5f, 0.0f },
	};

	// Heliar 100 mm (Tronnier; LensFlareFramework's heliar-tronnier). Its seventh surface is flat glass.
	constexpr LensRow kHeliar[] = {
		{ 30.81f, 7.7f, 1.652f, 29.0f, 58.57f },  { -89.35f, 1.85f, 1.603f, 29.0f, 38.03f },
		{ 580.38f, 3.52f, 1.0f, 29.0f, 0.0f },	  { -80.63f, 1.85f, 1.643f, 24.6f, 47.74f },
		{ 28.34f, 4.18f, 1.0f, 24.0f, 0.0f },	  { 0.0f, 3.0f, 0.0f, 23.2f, 0.0f },
		{ 0.0f, 1.85f, 1.581f, 24.6f, 40.98f },	  { 32.19f, 7.27f, 1.694f, 24.6f, 53.2f },
		{ -52.99f, 82.857f, 1.0f, 24.6f, 0.0f },
	};

	// Wide angle 22 mm (Nakamura; Smith, "Modern Lens Design" p. 360, as pbrt's wide.22mm). pbrt gives no Abbe
	// numbers: these are of catalogue glasses with the same indices.
	constexpr LensRow kWide22[] = {
		{ 35.98738f, 1.21638f, 1.54f, 23.716f, 59.7f },	  { 11.69718f, 9.9957f, 1.0f, 17.996f, 0.0f },
		{ 13.08714f, 5.12622f, 1.772f, 12.364f, 49.6f },  { -22.63294f, 1.76924f, 1.617f, 9.812f, 54.0f },
		{ 71.05802f, 0.8184f, 1.0f, 9.152f, 0.0f },		  { 0.0f, 2.27766f, 0.0f, 8.756f, 0.0f },
		{ -9.58584f, 2.43254f, 1.617f, 8.184f, 54.0f },	  { -11.28864f, 0.11506f, 1.0f, 9.152f, 0.0f },
		{ -166.7765f, 3.09606f, 1.713f, 10.648f, 53.8f }, { -7.5911f, 1.32682f, 1.805f, 11.44f, 25.4f },
		{ -16.7662f, 3.98068f, 1.0f, 12.276f, 0.0f },	  { -7.70286f, 1.21638f, 1.617f, 13.42f, 54.0f },
		{ -11.97328f, 0.0f, 1.0f, 17.996f, 0.0f },
	};

	// Fisheye 10 mm (Muller 16 mm f/4; Smith, "Modern Lens Design" p. 164, as pbrt's fisheye.10mm). Abbe numbers as
	// for kWide22.
	constexpr LensRow kFisheye10[] = {
		{ 30.2249f, 0.8335f, 1.62f, 30.34f, 60.3f }, { 11.3931f, 7.4136f, 1.0f, 20.68f, 0.0f },
		{ 75.2019f, 1.0654f, 1.639f, 17.8f, 55.4f }, { 8.3349f, 11.1549f, 1.0f, 13.42f, 0.0f },
		{ 9.5882f, 2.0054f, 1.654f, 9.02f, 39.7f },	 { 43.8677f, 5.3895f, 1.0f, 8.14f, 0.0f },
		{ 0.0f, 1.4163f, 0.0f, 6.08f, 0.0f },		 { 29.4541f, 2.1934f, 1.517f, 5.96f, 64.2f },
		{ -5.2265f, 0.9714f, 1.805f, 5.84f, 25.4f }, { -14.2884f, 0.0627f, 1.0f, 5.96f, 0.0f },
		{ -22.3726f, 0.94f, 1.673f, 5.96f, 32.1f },	 { -15.0404f, 0.0f, 1.0f, 6.52f, 0.0f },
	};

	// Nikon 28-75 mm zoom at its wide end (Grenier's PhysicallyBasedLensFlare, whose heights are semi-apertures).
	// Abbe numbers are of catalogue glasses with the same indices.
	constexpr LensRow kNikon28[] = {
		{ 72.747f, 2.3f, 1.60300f, 58.0f, 65.4f },		{ 37.0f, 13.0f, 1.0f, 58.0f, 0.0f },
		{ -172.809f, 2.1f, 1.58913f, 52.4f, 61.2f },	{ 39.894f, 1.0f, 1.0f, 52.4f, 0.0f },
		{ 49.82f, 4.4f, 1.86074f, 40.0f, 23.1f },		{ 74.75f, 53.142f, 1.0f, 40.0f, 0.0f },
		{ 63.402f, 1.6f, 1.86074f, 32.2f, 23.1f },		{ 37.53f, 8.6f, 1.51680f, 32.2f, 64.2f },
		{ -75.887f, 1.6f, 1.80458f, 32.0f, 25.5f },		{ -97.792f, 7.063f, 1.0f, 33.0f, 0.0f },
		{ 96.034f, 3.6f, 1.62041f, 36.0f, 60.3f },		{ 261.743f, 0.1f, 1.0f, 36.0f, 0.0f },
		{ 54.262f, 6.0f, 1.69680f, 36.0f, 55.5f },		{ -5995.277f, 1.532f, 1.0f, 36.0f, 0.0f },
		{ 0.0f, 2.8f, 0.0f, 14.0f, 0.0f },				{ -74.414f, 2.2f, 1.90265f, 26.0f, 35.7f },
		{ -62.929f, 1.45f, 1.51680f, 26.0f, 64.2f },	{ 121.38f, 2.5f, 1.0f, 26.2f, 0.0f },
		{ -85.723f, 1.4f, 1.49782f, 26.0f, 82.6f },		{ 31.093f, 2.6f, 1.80458f, 26.2f, 25.5f },
		{ 84.758f, 16.889f, 1.0f, 26.0f, 0.0f },		{ 459.69f, 1.4f, 1.86074f, 30.0f, 23.1f },
		{ 40.24f, 7.3f, 1.49782f, 30.0f, 82.6f },		{ -49.771f, 0.1f, 1.0f, 30.4f, 0.0f },
		{ 62.369f, 7.0f, 1.67025f, 32.0f, 57.6f },		{ -76.454f, 5.2f, 1.0f, 32.0f, 0.0f },
		{ -32.524f, 2.0f, 1.80454f, 34.0f, 39.6f },		{ -50.194f, 39.683f, 1.0f, 34.0f, 0.0f },
	};

	// The zooms of LensFlareFramework (canon-zoom-*, nikon-zoom-*, itoh-zoom-*): one prescription each, with the
	// gaps that move between the zoom positions patched in by LensDesign::gaps.
	constexpr LensRow kCanonZoom[] = {
		{ 311.919f, 2.8f, 1.7495f, 38.0f, 35.0f },	 { 118.63f, 0.42f, 1.0f, 38.0f, 0.0f },
		{ 128.135f, 8.68f, 1.497f, 38.0f, 81.6f },	 { -263.46f, 0.1f, 1.0f, 38.0f, 0.0f },
		{ 79.501f, 5.72f, 1.497f, 35.0f, 81.6f },	 { 203.191f, 8.78f, 1.0f, 35.0f, 0.0f },
		{ 54.391f, 2.2f, 1.847f, 28.0f, 23.8f },	 { 45.859f, 1.13f, 1.0f, 28.0f, 0.0f },
		{ 51.927f, 8.55f, 1.487f, 28.0f, 70.2f },	 { 5099.3f, 1.64f, 1.0f, 28.0f, 0.0f },
		{ -488.6f, 1.4f, 1.804f, 18.0f, 46.6f },	 { 35.39f, 5.88f, 1.0f, 18.0f, 0.0f },
		{ -78.125f, 1.4f, 1.487f, 18.0f, 70.2f },	 { 38.137f, 4.97f, 1.847f, 18.0f, 23.9f },
		{ 417.478f, 2.65f, 1.0f, 18.0f, 0.0f },		 { -66.802f, 1.4f, 1.729f, 18.0f, 54.7f },
		{ -3363.0f, 30.32f, 1.0f, 18.0f, 0.0f },	 { 247.12f, 3.49f, 1.697f, 19.0f, 55.5f },
		{ -99.902f, 0.15f, 1.0f, 19.0f, 0.0f },		 { -189.99f, 4.77f, 1.497f, 19.0f, 81.6f },
		{ -40.553f, 1.45f, 1.834f, 19.0f, 37.2f },	 { -76.277f, 14.7f, 1.0f, 19.0f, 0.0f },
		{ 58.421f, 3.53f, 1.804f, 21.0f, 46.6f },	 { 133.262f, 3.0f, 1.0f, 21.0f, 0.0f },
		{ 0.0f, 0.24f, 0.0f, 21.0f, 0.0f },			 { 34.132f, 6.35f, 1.497f, 21.0f, 81.6f },
		{ 2256.763f, 3.72f, 1.620f, 21.0f, 36.3f },	 { 31.519f, 28.17f, 1.0f, 21.0f, 0.0f },
		{ 132.947f, 5.9f, 1.517f, 21.0f, 52.4f },	 { -77.546f, 13.94f, 1.0f, 21.0f, 0.0f },
		{ -39.485f, 1.8f, 1.834f, 21.0f, 37.2f },	 { -95.683f, 0.15f, 1.0f, 21.0f, 0.0f },
		{ 147.644f, 3.62f, 1.743f, 21.0f, 49.3f },	 { -205.76f, 54.5f, 1.0f, 21.0f, 0.0f },
	};
	constexpr LensRow kNikonZoom[] = {
		{ 148.57f, 5.0f, 1.564f, 32.2f, 60.7f },	 { -479.279f, 0.357f, 1.0f, 32.2f, 0.0f },
		{ 192.855f, 6.286f, 1.589f, 31.0f, 61.2f },	 { -134.999f, 2.214f, 1.805f, 31.0f, 25.4f },
		{ -874.852f, 1.86f, 1.0f, 30.5f, 0.0f },	 { -285.712f, 3.893f, 1.755f, 19.0f, 27.5f },
		{ -44.764f, 1.036f, 1.517f, 19.0f, 64.2f },	 { 59.357f, 5.357f, 1.0f, 19.0f, 0.0f },
		{ -45.071f, 1.25f, 1.589f, 16.0f, 61.2f },	 { 166.174f, 36.928f, 1.0f, 16.0f, 0.0f },
		{ 164.284f, 5.0f, 1.658f, 15.0f, 57.3f },	 { -33.928f, 1.036f, 1.795f, 15.0f, 28.4f },
		{ -85.047f, 16.55f, 1.0f, 15.0f, 0.0f },	 { 30.464f, 5.357f, 1.670f, 16.7f, 57.5f },
		{ 162.202f, 3.071f, 1.0f, 16.3f, 0.0f },	 { -374.997f, 1.786f, 1.805f, 15.2f, 25.4f },
		{ 259.105f, 17.857f, 1.0f, 14.5f, 0.0f },	 { 0.0f, 18.0f, 0.0f, 9.4f, 0.0f },
		{ -17.214f, 1.714f, 1.713f, 10.5f, 53.8f },	 { -30.23f, 0.286f, 1.0f, 11.5f, 0.0f },
		{ 62.857f, 3.071f, 1.773f, 12.6f, 49.7f },	 { -466.815f, 40.08f, 1.0f, 12.6f, 0.0f },
	};
	constexpr LensRow kItohZoom[] = {
		{ 296.14f, 14.22f, 1.589f, 82.0f, 61.2f },	 { 914.14f, 0.41f, 1.0f, 82.0f, 0.0f },
		{ 187.78f, 4.88f, 1.806f, 75.0f, 40.9f },	 { 64.23f, 30.88f, 1.0f, 52.0f, 0.0f },
		{ 717.9f, 5.28f, 1.678f, 50.5f, 50.7f },	 { 120.23f, 18.81f, 1.0f, 50.5f, 0.0f },
		{ 113.65f, 12.19f, 1.805f, 42.0f, 25.4f },	 { 223.46f, 59.24f, 1.0f, 42.0f, 0.0f },
		{ 0.0f, 4.88f, 0.0f, 26.9f, 0.0f },			 { 171.17f, 28.32f, 1.806f, 35.0f, 40.9f },
		{ -565.34f, 9.26f, 1.0f, 35.0f, 0.0f },		 { 83.77f, 10.16f, 1.808f, 33.0f, 40.9f },
		{ 262.12f, 9.13f, 1.0f, 33.0f, 0.0f },		 { -218.78f, 15.93f, 1.847f, 32.0f, 23.9f },
		{ 85.25f, 7.31f, 1.0f, 32.0f, 0.0f },		 { -839.85f, 9.14f, 1.516f, 32.0f, 64.1f },
		{ -94.28f, 0.41f, 1.0f, 32.0f, 0.0f },		 { -3422.75f, 9.43f, 1.516f, 32.8f, 64.1f },
		{ -115.67f, 149.674f, 1.0f, 32.8f, 0.0f },
	};

	struct ZoomGap
	{
		int row = -1; // -1 = unused
		float thickness = 0.0f;
	};
	struct LensDesign
	{
		const char *name; // with the focal length and open f-number BuildLens finds, for the UI
		std::span<const LensRow> rows;
		std::array<ZoomGap, 4> gaps = {};
	};
	const LensDesign kLenses[] = {
		{ "Double Gauss 50 mm f/2.0 (11 surfaces)", kDoubleGauss },
		{ "Angenieux double Gauss 50 mm f/2.3 (15)", kAngenieux },
		{ "Heliar 99 mm f/3.4 (9)", kHeliar },
		{ "Wide angle 22 mm f/2.7 (13)", kWide22 },
		{ "Fisheye 10 mm f/4.0 (12)", kFisheye10 },
		{ "Nikon zoom 29 mm f/4.4 (28)", kNikon28 },
		{ "Canon zoom 72 mm f/4.8 (34)", kCanonZoom },
		{ "Canon zoom 135 mm f/4.8 (34)", kCanonZoom, { { { 5, 26.15f }, { 9, 12.89f }, { 16, 15.72f }, { 21, 0.69f } } } },
		{ "Canon zoom 194 mm f/4.8 (34)", kCanonZoom, { { { 5, 32.85f }, { 9, 17.23f }, { 16, 1.32f }, { 21, 4.05f } } } },
		{ "Nikon zoom 57 mm f/5.4 (22)", kNikonZoom },
		{ "Nikon zoom 93 mm f/5.9 (22)", kNikonZoom, { { { 4, 25.27f }, { 9, 19.23f }, { 12, 10.84f } } } },
		{ "Nikon zoom 143 mm f/5.4 (22)", kNikonZoom, { { { 4, 48.678f }, { 9, 1.536f }, { 12, 5.133f } } } },
		{ "Itoh zoom 100 mm f/6.9 (19)", kItohZoom },
		{ "Itoh zoom 119 mm f/7.4 (19)", kItohZoom, { { { 7, 33.205f } } } },
		{ "Itoh zoom 146 mm f/8.0 (19)", kItohZoom, { { { 7, 7.21f } } } },
	};

	// Design wavelengths (nm) of the quarter-wave coatings, cycled over a lens's air-glass surfaces. The real
	// coatings are unknown; spreading them over the spectrum gives the ghosts the varied tints of a real lens.
	// Cemented surfaces are not coated.
	constexpr float kCoatingPalette[] = { 520.0f, 600.0f, 460.0f, 570.0f, 490.0f, 620.0f, 440.0f, 550.0f };
	constexpr float kCoatingIor = 1.38f; // MgF2
	// LensFlare.slang's kWavelengths, per colour channel.
	constexpr float kChannelWavelengths[FLARE_WAVELENGTHS] = { 650.0f, 540.0f, 450.0f };
	// Ghosts dimmer than this fraction of the brightest are left out.
	constexpr float kMinGhostShare = 0.01f;

	constexpr uint32_t kSunSamples = FLARE_MAX_SAMPLES;
	constexpr uint32_t kEmitterSamples = 8;
	// LensFlare.slang's g_Args: the ghost and starburst draw arguments, then the light count.
	constexpr uint32_t kArgsCount = 9;
	// Half the height of the full-frame (36 x 24 mm) sensor the lens was designed for, in mm.
	constexpr float kSensorHalfHeight = 12.0f;
	// Exposed radiance below which a ghost quad is not drawn.
	constexpr float kMinVisible = 1.0e-3f;

	constexpr int kStarburstSize = 512;
	constexpr float kStarburstReferenceWavelength = 550.0f;

	// Refractive index at a wavelength from the d-line index and Abbe number, by Cauchy's two-term formula.
	float Dispersed(float ior, float abbe, float wavelengthNm)
	{
		if (abbe <= 0.0f || ior <= 1.0f)
		{
			return ior;
		}
		const double lF = 0.4861, lC = 0.6563, lD = 0.5876;
		const double b = (ior - 1.0) / (abbe * (1.0 / (lF * lF) - 1.0 / (lC * lC)));
		const double a = ior - b / (lD * lD);
		const double l = wavelengthNm * 1.0e-3;
		return static_cast<float>(a + b / (l * l));
	}

	// LensFlare.slang's CoatedReflectance at normal incidence.
	float CoatedReflectanceNormal(float wavelength, float thickness, float n0, float n1, float n2)
	{
		const double r01 = (n0 - n1) / static_cast<double>(n0 + n1);
		const double r12 = (n1 - n2) / static_cast<double>(n1 + n2);
		const double cosPhase = cos(4.0 * kPi * n1 * thickness / wavelength);
		return static_cast<float>((r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cosPhase) /
								  (1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * cosPhase));
	}

	struct DoubleRay
	{
		double p[3];
		double d[3];
	};

	double Dot3(const double a[3], const double b[3])
	{
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	}

	// LensFlare.slang's IntersectSurface and refraction, in double precision, for the first-order properties.
	bool TraceSurface(const GpuFlareSurface &surface, DoubleRay &ray, bool forward)
	{
		double normal[3] = { 0.0, 0.0, ray.d[2] > 0.0 ? -1.0 : 1.0 };
		if (surface.radius == 0.0f)
		{
			if (std::abs(ray.d[2]) <= 1.0e-12)
			{
				return false;
			}
			const double t = (surface.z - ray.p[2]) / ray.d[2];
			for (int i = 0; i < 3; ++i)
			{
				ray.p[i] += ray.d[i] * t;
			}
		}
		else
		{
			const double absRadius = std::abs(static_cast<double>(surface.radius));
			const double centre[3] = { 0.0, 0.0, static_cast<double>(surface.z) + surface.radius };
			const double oc[3] = { ray.p[0] - centre[0], ray.p[1] - centre[1], ray.p[2] - centre[2] };
			const double b = Dot3(ray.d, oc);
			const double c = Dot3(oc, oc) - absRadius * absRadius;
			const double discriminant = b * b - c;
			if (discriminant < 0.0)
			{
				return false;
			}
			const double root = sqrt(discriminant);
			const double t = surface.radius * ray.d[2] > 0.0 ? -b - root : -b + root;
			for (int i = 0; i < 3; ++i)
			{
				ray.p[i] += ray.d[i] * t;
			}
			for (int i = 0; i < 3; ++i)
			{
				normal[i] = (ray.p[i] - centre[i]) / absRadius;
			}
			if (Dot3(normal, ray.d) > 0.0)
			{
				for (double &n: normal)
				{
					n = -n;
				}
			}
		}
		if ((surface.flags & FLARE_SURFACE_STOP) != 0)
		{
			return true;
		}
		const double eta = forward ? surface.nBefore / static_cast<double>(surface.nAfter)
								   : surface.nAfter / static_cast<double>(surface.nBefore);
		const double cosI = -Dot3(ray.d, normal);
		const double k = 1.0 - eta * eta * (1.0 - cosI * cosI);
		if (k < 0.0)
		{
			return false;
		}
		for (int i = 0; i < 3; ++i)
		{
			ray.d[i] = eta * ray.d[i] + (eta * cosI - sqrt(k)) * normal[i];
		}
		return true;
	}

	constexpr float kBladeCurvature = 0.2f;

	float IrisDistance(float x, float y, int blades)
	{
		const float r = sqrtf(x * x + y * y);
		if (blades < 3)
		{
			return r;
		}
		const float sector = 2.0f * static_cast<float>(kPi) / blades;
		const float angle = atan2f(y, x);
		const float local = angle - sector * floorf(angle / sector) - 0.5f * sector;
		const float polygon = r * cosf(local) / cosf(0.5f * sector);
		return polygon + (r - polygon) * kBladeCurvature;
	}

	void Fft(std::complex<float> *data, int n, int stride, std::vector<std::complex<float>> &scratch)
	{
		scratch.resize(n);
		for (int i = 0; i < n; ++i)
		{
			scratch[i] = data[i * stride];
		}
		for (int i = 1, j = 0; i < n; ++i)
		{
			int bit = n >> 1;
			for (; j & bit; bit >>= 1)
			{
				j ^= bit;
			}
			j ^= bit;
			if (i < j)
			{
				std::swap(scratch[i], scratch[j]);
			}
		}
		for (int length = 2; length <= n; length <<= 1)
		{
			const double angle = -2.0 * kPi / length;
			const std::complex<float> step(static_cast<float>(cos(angle)), static_cast<float>(sin(angle)));
			for (int i = 0; i < n; i += length)
			{
				std::complex<float> w(1.0f, 0.0f);
				for (int j = 0; j < length / 2; ++j)
				{
					const std::complex<float> u = scratch[i + j];
					const std::complex<float> v = scratch[i + j + length / 2] * w;
					scratch[i + j] = u + v;
					scratch[i + j + length / 2] = u - v;
					w *= step;
				}
			}
		}
		for (int i = 0; i < n; ++i)
		{
			data[i * stride] = scratch[i];
		}
	}

	float3 WavelengthToRgb(float wavelength)
	{
		auto g = [wavelength](float mu, float sigma1, float sigma2)
		{
			const float t = (wavelength - mu) / (wavelength < mu ? sigma1 : sigma2);
			return expf(-0.5f * t * t);
		};
		const float x =
				1.056f * g(599.8f, 37.9f, 31.0f) + 0.362f * g(442.0f, 16.0f, 26.7f) - 0.065f * g(501.1f, 20.4f, 26.2f);
		const float y = 0.821f * g(568.8f, 46.9f, 40.5f) + 0.286f * g(530.9f, 16.3f, 31.1f);
		const float z = 1.217f * g(437.0f, 11.8f, 36.0f) + 0.681f * g(459.0f, 26.0f, 13.8f);
		return float3(std::max(3.2406f * x - 1.5372f * y - 0.4986f * z, 0.0f),
					  std::max(-0.9689f * x + 1.8758f * y + 0.0415f * z, 0.0f),
					  std::max(0.0557f * x - 0.2040f * y + 1.0570f * z, 0.0f));
	}

	float Frac(float x)
	{
		return x - floorf(x);
	}

	uint32_t FindRoot(std::vector<uint32_t> &parent, uint32_t i)
	{
		while (parent[i] != i)
		{
			parent[i] = parent[parent[i]];
			i = parent[i];
		}
		return i;
	}

	void Unite(std::vector<uint32_t> &parent, uint32_t a, uint32_t b)
	{
		a = FindRoot(parent, a);
		b = FindRoot(parent, b);
		if (a != b)
		{
			parent[std::max(a, b)] = std::min(a, b);
		}
	}

	// Splits an emissive instance's triangle lights (indices into lights) into its separate pieces, so that each bulb
	// of a mesh holding many gets a flare of its own at its own direction rather than one at their mean. Triangles
	// sharing a vertex position join, then pieces whose bounds overlap join (a bulb's shell and its filament).
	std::vector<std::vector<uint32_t>> SplitEmitter(std::span<const GpuLight> lights, const std::vector<uint32_t> &members)
	{
		const uint32_t count = static_cast<uint32_t>(members.size());
		std::vector<uint32_t> parent(count);
		std::iota(parent.begin(), parent.end(), 0u);

		struct Key
		{
			uint32_t x, y, z;
			bool operator==(const Key &) const = default;
		};
		struct KeyHash
		{
			size_t operator()(const Key &k) const
			{
				return (static_cast<size_t>(k.x) * 73856093u) ^ (static_cast<size_t>(k.y) * 19349663u) ^
					   (static_cast<size_t>(k.z) * 83492791u);
			}
		};
		auto bits = [](float v)
		{
			v += 0.0f; // -0 as +0
			uint32_t b;
			std::memcpy(&b, &v, sizeof(b));
			return b;
		};
		std::unordered_map<Key, uint32_t, KeyHash> firstAt;
		firstAt.reserve(static_cast<size_t>(count) * 3);
		for (uint32_t t = 0; t < count; ++t)
		{
			const GpuLight &light = lights[members[t]];
			for (const float4 *v: { &light.v1, &light.v2, &light.v3 })
			{
				const auto [it, inserted] = firstAt.try_emplace(Key{ bits(v->x), bits(v->y), bits(v->z) }, t);
				if (!inserted)
				{
					Unite(parent, t, it->second);
				}
			}
		}

		// The connected pieces' bounds, then a sweep along x joining pieces whose bounds touch.
		struct Box
		{
			float lo[3];
			float hi[3];
			uint32_t piece;
		};
		std::vector<Box> boxes;
		std::vector<uint32_t> boxOf(count, UINT32_MAX);
		for (uint32_t t = 0; t < count; ++t)
		{
			const uint32_t root = FindRoot(parent, t);
			if (boxOf[root] == UINT32_MAX)
			{
				boxOf[root] = static_cast<uint32_t>(boxes.size());
				boxes.push_back({ { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX }, root });
			}
			Box &box = boxes[boxOf[root]];
			const GpuLight &light = lights[members[t]];
			for (const float4 *v: { &light.v1, &light.v2, &light.v3 })
			{
				const float p[3] = { v->x, v->y, v->z };
				for (int a = 0; a < 3; ++a)
				{
					box.lo[a] = std::min(box.lo[a], p[a]);
					box.hi[a] = std::max(box.hi[a], p[a]);
				}
			}
		}
		std::sort(boxes.begin(), boxes.end(), [](const Box &a, const Box &b) { return a.lo[0] < b.lo[0]; });
		for (size_t i = 0; i < boxes.size(); ++i)
		{
			for (size_t j = i + 1; j < boxes.size() && boxes[j].lo[0] <= boxes[i].hi[0]; ++j)
			{
				if (boxes[j].lo[1] <= boxes[i].hi[1] && boxes[i].lo[1] <= boxes[j].hi[1] &&
					boxes[j].lo[2] <= boxes[i].hi[2] && boxes[i].lo[2] <= boxes[j].hi[2])
				{
					Unite(parent, boxes[i].piece, boxes[j].piece);
				}
			}
		}

		std::vector<std::vector<uint32_t>> pieces;
		std::vector<uint32_t> pieceOf(count, UINT32_MAX);
		for (uint32_t t = 0; t < count; ++t)
		{
			const uint32_t root = FindRoot(parent, t);
			if (pieceOf[root] == UINT32_MAX)
			{
				pieceOf[root] = static_cast<uint32_t>(pieces.size());
				pieces.emplace_back();
			}
			pieces[pieceOf[root]].push_back(t);
		}
		return pieces;
	}
} // namespace

LensFlarePass::LensFlarePass(RenderContext *renderContext,
							 const RenderGPUScene &scene,
							 const BloomPass &bloom,
							 ResourceHandle exposure,
							 const Camera &camera,
							 const RenderSettings &settings) :
	renderContext_(renderContext), scene_(scene), bloom_(bloom), exposure_(exposure), camera_(camera),
	settings_(settings)
{
	if (renderContext_ == nullptr)
	{
		throw std::invalid_argument("LensFlarePass requires a valid RenderContext.");
	}

	auto computePipeline = [&](const char *entryPoint)
	{
		const PipelineHandle pipeline =
				renderContext_->CreateComputePipeline(PATHTRACER_SHADER_DIR "LensFlare.slang", entryPoint);
		if (pipeline == InvalidPipelineHandle)
		{
			throw std::runtime_error(std::string("Failed to create the lens flare pipeline ") + entryPoint +
									 " (see the shader log above).");
		}
		return pipeline;
	};
	selectPipeline_ = computePipeline("Select");
	tracePipeline_ = computePipeline("Trace");
	compositePipeline_ = computePipeline("Composite");
	ghostPipeline_ = CreateGraphicsPipeline("GhostVS", "GhostPS");
	starburstPipeline_ = CreateGraphicsPipeline("StarburstVS", "StarburstPS");
	drawSignature_ = renderContext_->CreateCommandSignature(IndirectCommandType::Draw, 0, InvalidPipelineHandle);

	width_ = renderContext_->GetWindowWidth();
	height_ = renderContext_->GetWindowHeight();
	flareWidth_ = std::max((width_ + 1) / 2, 1u);
	flareHeight_ = std::max((height_ + 1) / 2, 1u);

	lens_ = std::clamp(settings_.lensFlare.lens, 0, GetLensCount() - 1);
	BuildLens(lens_);
	BuildCandidates();

	auto createBuffer = [&](uint64_t count, uint32_t stride, const char *debugName)
	{
		BufferDesc desc = {};
		desc.sizeInBytes = std::max<uint64_t>(count, 1) * stride;
		desc.structureByteStride = stride;
		desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
		desc.debugName = debugName;
		return renderContext_->CreateBuffer(desc, ResourceViewType::UnorderedAccess);
	};

	// Sized for any lens, so switching lenses never reallocates them.
	const uint64_t instances = static_cast<uint64_t>(FLARE_MAX_LIGHTS) * FLARE_MAX_GHOSTS * FLARE_WAVELENGTHS;
	const uint64_t totalQuads = instances * FLARE_GRID * FLARE_GRID;
	if (totalQuads > std::numeric_limits<uint32_t>::max())
	{
		throw std::overflow_error("LensFlarePass quad capacity exceeds uint32_t.");
	}
	quadCapacity_ = static_cast<uint32_t>(totalQuads);

	results_ = createBuffer(candidateCount_, sizeof(GpuFlareResult), "LensFlare_Results");
	lights_ = createBuffer(FLARE_MAX_LIGHTS, sizeof(GpuFlareLight), "LensFlare_Lights");
	args_ = createBuffer(kArgsCount, sizeof(uint32_t), "LensFlare_Args");
	ghostVertices_ = createBuffer(instances * FLARE_GRID_VERTICES, sizeof(float4), "LensFlare_GhostVertices");
	ghostApertures_ = createBuffer(instances * FLARE_GRID_VERTICES, sizeof(uint32_t), "LensFlare_GhostApertures");
	quads_ = createBuffer(quadCapacity_, sizeof(uint32_t), "LensFlare_Quads");

	constantBuffers_.resize(renderContext_->GetFramesInFlight());
	for (ResourceHandle &buffer: constantBuffers_)
	{
		buffer = renderContext_->CreateConstantBuffer(sizeof(LensFlareConstants), "LensFlare_Constants");
	}
}

LensFlarePass::~LensFlarePass()
{
	for (const Retired &retired: retired_)
	{
		renderContext_->DestroyResource(retired.resource);
	}
	for (ResourceHandle buffer: { surfaceBuffer_,
								  ghostBuffer_,
								  candidates_,
								  samples_,
								  results_,
								  lights_,
								  args_,
								  ghostVertices_,
								  ghostApertures_,
								  quads_,
								  starburst_ })
	{
		if (buffer != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(buffer);
		}
	}
	for (ResourceHandle buffer: constantBuffers_)
	{
		if (buffer != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(buffer);
		}
	}
}

PipelineHandle LensFlarePass::CreateGraphicsPipeline(const char *vertexEntry, const char *pixelEntry) const
{
	PipelineEntryPoint entries[] = { { vertexEntry, PipelineShaderStage::Vertex },
									 { pixelEntry, PipelineShaderStage::Pixel } };
	GraphicsPipelineCreateInfo info;
	info.rasterizerState.cullMode = CullMode::None;
	info.depthStencilState.depthEnable = false;
	info.depthStencilState.depthWrite = false;
	info.blendState.renderTarget.blendEnable = true;
	info.blendState.renderTarget.srcBlend = BlendFactor::One;
	info.blendState.renderTarget.destBlend = BlendFactor::One;
	info.blendState.renderTarget.srcBlendAlpha = BlendFactor::One;
	info.blendState.renderTarget.destBlendAlpha = BlendFactor::One;
	info.renderTargetFormats = { Format::R16G16B16A16_FLOAT };
	const PipelineHandle pipeline =
			renderContext_->CreateGraphicsPipeline(PATHTRACER_SHADER_DIR "LensFlare.slang", entries, info);
	if (pipeline == InvalidPipelineHandle)
	{
		throw std::runtime_error(std::string("Failed to create the lens flare pipeline ") + vertexEntry +
								 " (see the shader log above).");
	}
	return pipeline;
}

int LensFlarePass::GetLensCount()
{
	return static_cast<int>(std::size(kLenses));
}

const char *LensFlarePass::GetLensName(int lens)
{
	return lens >= 0 && lens < GetLensCount() ? kLenses[lens].name : "";
}

void LensFlarePass::BuildLens(int lens)
{
	lens = std::clamp(lens, 0, GetLensCount() - 1);
	const LensDesign &design = kLenses[lens];
	std::vector<LensRow> rows(design.rows.begin(), design.rows.end());
	for (const ZoomGap &gap: design.gaps)
	{
		if (gap.row >= 0 && gap.row < static_cast<int>(rows.size()))
		{
			rows[gap.row].thickness = gap.thickness;
		}
	}
	if (rows.size() > FLARE_MAX_SURFACES)
	{
		throw std::runtime_error("Lens flare: the lens has more than FLARE_MAX_SURFACES surfaces.");
	}

	surfaces_.clear();
	ghosts_.clear();
	stopSurface_ = UINT32_MAX;

	float z = 0.0f;
	float nBefore = 1.0f;
	std::vector<float> abbeAfter;
	uint32_t coated = 0;
	for (size_t i = 0; i < rows.size(); ++i)
	{
		const LensRow &row = rows[i];
		const bool stop = row.ior == 0.0f;
		GpuFlareSurface surface = {};
		surface.z = z;
		surface.radius = stop ? 0.0f : row.radius;
		surface.semiAperture = 0.5f * row.diameter;
		surface.nBefore = nBefore;
		surface.nAfter = stop ? nBefore : row.ior;
		if (stop)
		{
			surface.flags = FLARE_SURFACE_STOP;
			stopSurface_ = static_cast<uint32_t>(i);
			openStopRadius_ = surface.semiAperture;
		}
		const bool airGlass = !stop && (surface.nBefore == 1.0f || surface.nAfter == 1.0f);
		surface.coatingIor = airGlass ? std::max(sqrtf(surface.nBefore * surface.nAfter), kCoatingIor) : surface.nAfter;
		surface.coatingThickness = 0.0f;
		if (airGlass)
		{
			const float wavelength = kCoatingPalette[coated++ % std::size(kCoatingPalette)];
			surface.coatingThickness = wavelength / (4.0f * surface.coatingIor);
		}
		surfaces_.push_back(surface);
		abbeAfter.push_back(stop ? (abbeAfter.empty() ? 0.0f : abbeAfter.back()) : row.abbe);
		nBefore = surface.nAfter;
		z += row.thickness;
	}
	if (stopSurface_ == UINT32_MAX)
	{
		throw std::runtime_error("Lens flare: the lens has no aperture stop.");
	}

	const double h = 0.1;
	DoubleRay marginal = { { 0.0, h, surfaces_[0].z - 1.0 }, { 0.0, 0.0, 1.0 } };
	double stopHeight = 0.0;
	for (size_t i = 0; i < surfaces_.size(); ++i)
	{
		if (!TraceSurface(surfaces_[i], marginal, true))
		{
			throw std::runtime_error("Lens flare: the lens prescription does not trace.");
		}
		if (i == stopSurface_)
		{
			stopHeight = marginal.p[1];
		}
	}
	if (std::abs(marginal.d[1]) <= 1.0e-12 || std::abs(stopHeight) <= 1.0e-12)
	{
		throw std::runtime_error("Lens flare: degenerate paraxial marginal ray.");
	}
	sensorZ_ = static_cast<float>(marginal.p[2] - marginal.p[1] * marginal.d[2] / marginal.d[1]);
	focalLength_ = static_cast<float>(h * marginal.d[2] / -marginal.d[1]);
	const double entrancePupilRadius = openStopRadius_ * h / stopHeight;
	openFNumber_ = static_cast<float>(focalLength_ / (2.0 * entrancePupilRadius));

	const double slope = 0.01;
	DoubleRay chief = { { 0.0, 0.0, surfaces_[stopSurface_].z },
						{ 0.0, slope / sqrt(1.0 + slope * slope), -1.0 / sqrt(1.0 + slope * slope) } };
	for (int i = static_cast<int>(stopSurface_) - 1; i >= 0; --i)
	{
		if (!TraceSurface(surfaces_[i], chief, false))
		{
			throw std::runtime_error("Lens flare: the lens prescription does not trace.");
		}
	}
	if (std::abs(chief.d[1]) <= 1.0e-12)
	{
		throw std::runtime_error("Lens flare: degenerate paraxial chief ray.");
	}
	entranceZ_ = static_cast<float>(chief.p[2] - chief.p[1] * chief.d[2] / chief.d[1]);
	entranceRadius_ = 1.05f * std::max(static_cast<float>(entrancePupilRadius), surfaces_[0].semiAperture);

	struct Candidate
	{
		GpuFlareGhost ghost;
		float strength;
	};
	std::vector<Candidate> candidates;
	auto reflectance = [&](uint32_t i)
	{
		const GpuFlareSurface &s = surfaces_[i];
		float sum = 0.0f;
		for (float wavelength: kChannelWavelengths)
		{
			sum += CoatedReflectanceNormal(wavelength, s.coatingThickness, s.nBefore, s.coatingIor, s.nAfter);
		}
		return sum / FLARE_WAVELENGTHS;
	};
	for (uint32_t first = 0; first < surfaces_.size(); ++first)
	{
		for (uint32_t second = 0; second < first; ++second)
		{
			if (first == stopSurface_ || second == stopSurface_ || (first < stopSurface_) != (second < stopSurface_))
			{
				continue;
			}
			candidates.push_back({ { first, second, 0.0f, 0.0f }, reflectance(first) * reflectance(second) });
		}
	}
	std::sort(candidates.begin(),
			  candidates.end(),
			  [](const Candidate &a, const Candidate &b) { return a.strength > b.strength; });
	const float brightest = candidates.empty() ? 0.0f : candidates.front().strength;
	for (const Candidate &candidate: candidates)
	{
		if (ghosts_.size() < FLARE_MAX_GHOSTS && candidate.strength >= kMinGhostShare * brightest)
		{
			ghosts_.push_back(candidate.ghost);
		}
	}

	std::vector<GpuFlareSurface> dispersed;
	dispersed.reserve(surfaces_.size() * FLARE_WAVELENGTHS);
	for (float wavelength: kChannelWavelengths)
	{
		float abbeBefore = 0.0f;
		for (size_t i = 0; i < surfaces_.size(); ++i)
		{
			GpuFlareSurface surface = surfaces_[i];
			surface.nBefore = Dispersed(surface.nBefore, abbeBefore, wavelength);
			surface.nAfter = Dispersed(surface.nAfter, abbeAfter[i], wavelength);
			dispersed.push_back(surface);
			abbeBefore = abbeAfter[i];
		}
	}

	const UploadBatchHandle batch = renderContext_->BeginUpload();
	BufferDesc desc = {};
	desc.sizeInBytes = dispersed.size() * sizeof(GpuFlareSurface);
	desc.structureByteStride = sizeof(GpuFlareSurface);
	desc.debugName = "LensFlare_Surfaces";
	surfaceBuffer_ = renderContext_->CreateBuffer(desc, ResourceViewType::ShaderResource, dispersed.data(), batch);
	desc.sizeInBytes = std::max<size_t>(ghosts_.size(), 1) * sizeof(GpuFlareGhost);
	desc.structureByteStride = sizeof(GpuFlareGhost);
	desc.debugName = "LensFlare_Ghosts";
	const GpuFlareGhost none = {};
	ghostBuffer_ = renderContext_->CreateBuffer(desc,
												ResourceViewType::ShaderResource,
												ghosts_.empty() ? &none : ghosts_.data(),
												batch);
	renderContext_->EndUpload(batch);
}

void LensFlarePass::BuildCandidates()
{
	std::vector<GpuFlareCandidate> candidates;
	std::vector<GpuFlareSample> samples;
	auto endCandidate = [&](uint32_t firstSample)
	{ candidates.push_back({ firstSample, static_cast<uint32_t>(samples.size()) - firstSample, 0, 0 }); };

	{
		const uint32_t first = static_cast<uint32_t>(samples.size());
		for (uint32_t k = 0; k < kSunSamples; ++k)
		{
			samples.push_back(
					{ FLARE_SUN_LIGHT, 1.0f / kSunSamples, float2((k + 0.5f) / kSunSamples, Frac(k * 0.6180339887f)) });
		}
		endCandidate(first);
	}

	const uint32_t triangleLights = scene_.GetTriangleLightCount();
	for (uint32_t i = 0; i < scene_.GetPunctualLightCount() && candidates.size() < FLARE_MAX_CANDIDATES; ++i)
	{
		const uint32_t first = static_cast<uint32_t>(samples.size());
		samples.push_back({ triangleLights + i, 1.0f, float2(0.0f, 0.0f) });
		endCandidate(first);
	}

	const std::span<const GpuLight> lights = scene_.GetTriangleLights();
	const std::span<const float> fluxScales = scene_.GetTriangleLightFluxScales();
	const std::span<const uint32_t> slots = scene_.GetLightEmitterSlots();
	struct Emitter
	{
		std::vector<uint32_t> lights;
		std::vector<float> flux;
		double total = 0.0;
	};
	std::vector<Emitter> emitters;
	for (uint32_t light = 0; light < lights.size() && light < slots.size(); ++light)
	{
		const float flux = LightTreeFlux(lights[light], light < fluxScales.size() ? fluxScales[light] : 1.0f);
		if (!(flux > 0.0f) || !std::isfinite(flux))
		{
			continue;
		}
		const uint32_t slot = slots[light];
		if (slot >= FLARE_MAX_CANDIDATES * 4)
		{
			continue;
		}
		if (slot >= emitters.size())
		{
			emitters.resize(static_cast<size_t>(slot) + 1);
		}
		Emitter &emitter = emitters[slot];
		emitter.lights.push_back(light);
		emitter.flux.push_back(flux);
		emitter.total += flux;
	}
	// A candidate per separate piece of each instance (one per bulb of a string of lights), so a flare sits on its bulb.
	{
		std::vector<Emitter> pieces;
		for (const Emitter &emitter: emitters)
		{
			if (emitter.lights.empty())
			{
				continue;
			}
			for (const std::vector<uint32_t> &members: SplitEmitter(lights, emitter.lights))
			{
				Emitter &piece = pieces.emplace_back();
				for (const uint32_t t: members)
				{
					piece.lights.push_back(emitter.lights[t]);
					piece.flux.push_back(emitter.flux[t]);
					piece.total += emitter.flux[t];
				}
			}
		}
		emitters = std::move(pieces);
	}
	std::vector<uint32_t> order(emitters.size());
	std::iota(order.begin(), order.end(), 0u);
	std::sort(order.begin(),
			  order.end(),
			  [&](uint32_t a, uint32_t b) { return emitters[a].total > emitters[b].total; });
	for (uint32_t index: order)
	{
		const Emitter &emitter = emitters[index];
		if (emitter.total <= 0.0 || candidates.size() >= FLARE_MAX_CANDIDATES)
		{
			break;
		}
		const uint32_t first = static_cast<uint32_t>(samples.size());
		for (uint32_t k = 0; k < kEmitterSamples; ++k)
		{
			const double target = (k + 0.5) / kEmitterSamples * emitter.total;
			double sum = 0.0;
			size_t triangle = 0;
			for (; triangle + 1 < emitter.flux.size(); ++triangle)
			{
				sum += emitter.flux[triangle];
				if (sum >= target)
				{
					break;
				}
			}
			const double probability = emitter.flux[triangle] / emitter.total;
			samples.push_back({ emitter.lights[triangle],
								static_cast<float>(1.0 / (kEmitterSamples * probability)),
								float2(Frac(0.5f + k * 0.7548776662f), Frac(0.5f + k * 0.5698402910f)) });
		}
		endCandidate(first);
	}

	candidateCount_ = static_cast<uint32_t>(candidates.size());
	const UploadBatchHandle batch = renderContext_->BeginUpload();
	BufferDesc desc = {};
	desc.sizeInBytes = std::max<size_t>(candidates.size(), 1) * sizeof(GpuFlareCandidate);
	desc.structureByteStride = sizeof(GpuFlareCandidate);
	desc.debugName = "LensFlare_Candidates";
	candidates_ = renderContext_->CreateBuffer(desc, ResourceViewType::ShaderResource, candidates.data(), batch);
	desc.sizeInBytes = std::max<size_t>(samples.size(), 1) * sizeof(GpuFlareSample);
	desc.structureByteStride = sizeof(GpuFlareSample);
	desc.debugName = "LensFlare_Samples";
	samples_ = renderContext_->CreateBuffer(desc, ResourceViewType::ShaderResource, samples.data(), batch);
	renderContext_->EndUpload(batch);
}

void LensFlarePass::BuildStarburst(int blades)
{
	const int n = kStarburstSize;
	const float radius = n / 8.0f;
	std::vector<std::complex<float>> field(static_cast<size_t>(n) * n);
	for (int y = 0; y < n; ++y)
	{
		for (int x = 0; x < n; ++x)
		{
			const float px = (x + 0.5f - 0.5f * n) / radius;
			const float py = -(y + 0.5f - 0.5f * n) / radius;
			const float edge = (1.0f - IrisDistance(px, py, blades)) * radius;
			field[static_cast<size_t>(y) * n + x] = std::clamp(edge + 0.5f, 0.0f, 1.0f);
		}
	}

	std::vector<std::complex<float>> scratch;
	for (int y = 0; y < n; ++y)
	{
		Fft(&field[static_cast<size_t>(y) * n], n, 1, scratch);
	}

	for (int x = 0; x < n; ++x)
	{
		Fft(&field[x], n, n, scratch);
	}
	std::vector<float> power(static_cast<size_t>(n) * n);
	for (int y = 0; y < n; ++y)
	{
		for (int x = 0; x < n; ++x)
		{
			power[static_cast<size_t>(y) * n + x] =
					std::norm(field[static_cast<size_t>((y + n / 2) % n) * n + (x + n / 2) % n]);
		}
	}
	
	auto sample = [&](float x, float y)
	{
		x = std::clamp(x, 0.0f, n - 1.001f);
		y = std::clamp(y, 0.0f, n - 1.001f);
		const int x0 = static_cast<int>(x);
		const int y0 = static_cast<int>(y);
		const float fx = x - x0;
		const float fy = y - y0;
		const float *row0 = &power[static_cast<size_t>(y0) * n];
		const float *row1 = row0 + n;
		return (row0[x0] * (1.0f - fx) + row0[x0 + 1] * fx) * (1.0f - fy) +
			   (row1[x0] * (1.0f - fx) + row1[x0 + 1] * fx) * fy;
	};

	std::vector<float4> texels(static_cast<size_t>(n) * n, float4(0.0f, 0.0f, 0.0f, 0.0f));
	const float centre = 0.5f * n;
	for (float wavelength = 400.0f; wavelength <= 700.0f; wavelength += 5.0f)
	{
		const float3 colour = WavelengthToRgb(wavelength);
		const float scale = kStarburstReferenceWavelength / wavelength;
		const float energy = scale * scale;
		for (int y = 0; y < n; ++y)
		{
			for (int x = 0; x < n; ++x)
			{
				const float value = sample((x - centre) * scale + centre, (y - centre) * scale + centre) * energy;
				float4 &texel = texels[static_cast<size_t>(y) * n + x];
				texel.x += colour.x * value;
				texel.y += colour.y * value;
				texel.z += colour.z * value;
			}
		}
	}

	{
		auto sampleTexel = [&](float x, float y)
		{
			if (x < 0.0f || y < 0.0f || x > n - 1.001f || y > n - 1.001f)
			{
				return float4(0.0f, 0.0f, 0.0f, 0.0f);
			}
			const int x0 = static_cast<int>(x);
			const int y0 = static_cast<int>(y);
			const float fx = x - x0;
			const float fy = y - y0;
			const float4 *row0 = &texels[static_cast<size_t>(y0) * n];
			const float4 *row1 = row0 + n;
			const float w00 = (1.0f - fx) * (1.0f - fy);
			const float w10 = fx * (1.0f - fy);
			const float w01 = (1.0f - fx) * fy;
			const float w11 = fx * fy;
			return float4(row0[x0].x * w00 + row0[x0 + 1].x * w10 + row1[x0].x * w01 + row1[x0 + 1].x * w11,
						  row0[x0].y * w00 + row0[x0 + 1].y * w10 + row1[x0].y * w01 + row1[x0 + 1].y * w11,
						  row0[x0].z * w00 + row0[x0 + 1].z * w10 + row1[x0].z * w01 + row1[x0 + 1].z * w11,
						  0.0f);
		};
		constexpr int kRotations = 24;
		constexpr float kSpread = 0.03f;
		std::vector<float4> blurred(texels.size(), float4(0.0f, 0.0f, 0.0f, 0.0f));
		for (int i = 0; i < kRotations; ++i)
		{
			const float angle = kSpread * ((i + 0.5f) / kRotations * 2.0f - 1.0f);
			const float c = cosf(angle);
			const float s = sinf(angle);
			for (int y = 0; y < n; ++y)
			{
				for (int x = 0; x < n; ++x)
				{
					const float dx = x - centre;
					const float dy = y - centre;
					const float4 value = sampleTexel(centre + dx * c - dy * s, centre + dx * s + dy * c);
					float4 &texel = blurred[static_cast<size_t>(y) * n + x];
					texel.x += value.x / kRotations;
					texel.y += value.y / kRotations;
					texel.z += value.z / kRotations;
				}
			}
		}
		texels.swap(blurred);
	}

	const float airyRadius = 0.61f * n / radius;
	double sums[3] = {};
	for (int y = 0; y < n; ++y)
	{
		for (int x = 0; x < n; ++x)
		{
			const float dx = (x + 0.5f - centre) / centre;
			const float dy = (y + 0.5f - centre) / centre;
			const float r = sqrtf(dx * dx + dy * dy);
			const float t = std::clamp((r - 0.7f) / 0.3f, 0.0f, 1.0f);
			const float core = std::clamp((r * centre - airyRadius) / (2.0f * airyRadius), 0.0f, 1.0f);
			const float window = (1.0f - t * t * (3.0f - 2.0f * t)) * core * core * (3.0f - 2.0f * core) * r;
			float4 &texel = texels[static_cast<size_t>(y) * n + x];
			texel.x *= window;
			texel.y *= window;
			texel.z *= window;
			sums[0] += texel.x;
			sums[1] += texel.y;
			sums[2] += texel.z;
		}
	}
	const double texelCount = static_cast<double>(n) * n;
	const float scales[3] = { static_cast<float>(texelCount / std::max(sums[0], 1.0e-30)),
							  static_cast<float>(texelCount / std::max(sums[1], 1.0e-30)),
							  static_cast<float>(texelCount / std::max(sums[2], 1.0e-30)) };
	for (float4 &texel: texels)
	{
		texel = float4(texel.x * scales[0], texel.y * scales[1], texel.z * scales[2], 1.0f);
	}

	if (starburst_ != InvalidResourceHandle)
	{
		retired_.push_back({ starburst_, frameCount_ });
	}
	TextureDesc desc = {};
	desc.width = n;
	desc.height = n;
	desc.format = Format::R32G32B32A32_FLOAT;
	desc.debugName = "LensFlare_Starburst";
	TextureMipData mip = {};
	mip.pixels = texels.data();
	mip.rowPitch = n * sizeof(float4);
	mip.rowCount = n;
	const UploadBatchHandle batch = renderContext_->BeginUpload();
	starburst_ = renderContext_->CreateTexture(desc,
											   ResourceViewType::ShaderResource,
											   std::span<const TextureMipData>(&mip, 1),
											   batch);
	renderContext_->EndUpload(batch);
	starburstBlades_ = blades;
}

PathTracerPass::FlareVisibility LensFlarePass::GetVisibility() const
{
	PathTracerPass::FlareVisibility visibility;
	visibility.candidates = candidates_;
	visibility.samples = samples_;
	visibility.results = results_;
	visibility.candidateCount = candidateCount_;
	return visibility;
}

void LensFlarePass::AddPasses(FrameGraph &frame)
{
	++frameCount_;
	const uint64_t framesInFlight = renderContext_->GetFramesInFlight();
	std::erase_if(retired_,
				  [&](const Retired &retired)
				  {
					  if (retired.frame + framesInFlight + 1 > frameCount_)
					  {
						  return false;
					  }
					  renderContext_->DestroyResource(retired.resource);
					  return true;
				  });

	const LensFlareSettings &flare = settings_.lensFlare;
	const int lens = std::clamp(flare.lens, 0, GetLensCount() - 1);
	if (lens != lens_)
	{
		// The old lens's buffers may still be read by frames in flight.
		retired_.push_back({ surfaceBuffer_, frameCount_ });
		retired_.push_back({ ghostBuffer_, frameCount_ });
		surfaceBuffer_ = InvalidResourceHandle;
		ghostBuffer_ = InvalidResourceHandle;
		lens_ = lens;
		BuildLens(lens_);
	}

	if (!flare.enabled || candidateCount_ == 0 || ghosts_.empty() || !scene_.HasGeometry())
	{
		return;
	}

	const uint32_t currentWidth = renderContext_->GetWindowWidth();
	const uint32_t currentHeight = renderContext_->GetWindowHeight();
	if (currentWidth > 0 && currentHeight > 0 && (currentWidth != width_ || currentHeight != height_))
	{
		width_ = currentWidth;
		height_ = currentHeight;
		flareWidth_ = std::max((width_ + 1) / 2, 1u);
		flareHeight_ = std::max((height_ + 1) / 2, 1u);
	}

	const int blades = flare.apertureBlades >= 3 ? std::min(flare.apertureBlades, 32) : 0;
	if (blades != starburstBlades_)
	{
		BuildStarburst(blades);
	}

	LensFlareConstants constants = {};
	const float4x4 projection = camera_.GetProjectionMatrix();
	constants.tanHalfFovX = 1.0f / std::max(std::abs(projection._11), 1.0e-5f);
	constants.tanHalfFovY = 1.0f / std::max(std::abs(projection._22), 1.0e-5f);
	constants.cameraRight = camera_.GetRight();
	constants.cameraUp = camera_.GetUp();
	constants.cameraFront = camera_.GetFront();
	constants.threshold = std::max(flare.minIntensity, 0.0f);
	constants.compression = std::clamp(flare.compression, 0.0f, 1.0f);
	constants.candidateCount = candidateCount_;
	constants.maxLights = static_cast<uint32_t>(std::clamp(flare.maxLights, 1, FLARE_MAX_LIGHTS));
	constants.ghostCount = static_cast<uint32_t>(ghosts_.size());
	constants.surfaceCount = static_cast<uint32_t>(surfaces_.size());
	constants.entranceZ = entranceZ_;
	constants.entranceRadius = entranceRadius_;
	constants.sensorZ = sensorZ_;
	constants.focalLength = focalLength_;
	const float fNumber = std::max(flare.fNumber, openFNumber_);
	const float pupilRadius = focalLength_ / (2.0f * fNumber);
	const float polygonShare = blades >= 3 ? static_cast<float>(blades / (2.0 * kPi) * sin(2.0 * kPi / blades)) : 1.0f;
	constants.pupilArea = static_cast<float>(kPi) * pupilRadius * pupilRadius * polygonShare;
	constants.stopRadius = openStopRadius_ * openFNumber_ / fNumber;
	constants.ghostIntensity = std::max(flare.ghostIntensity, 0.0f);
	constants.starburstIntensity = std::max(flare.starburstIntensity, 0.0f);
	constants.starburstHalfSize = std::max(flare.starburstSize, 1.0e-3f) * constants.tanHalfFovY * fNumber / 8.0f;
	constants.blades = static_cast<uint32_t>(blades);
	constants.bladeRotation = 0.0f;
	constants.minVisible = kMinVisible;
	constants.outputWidth = width_;
	constants.outputHeight = height_;
	constants.flareTexelSize = float2(1.0f / flareWidth_, 1.0f / flareHeight_);
	constants.quadCapacity = quadCapacity_;
	constants.ghostSoftness = std::max(flare.ghostSoftness, 0.0f);
	// The screen is the lens's own frame: its height spans the full-frame sensor's, whatever the camera's field of view.
	// Mapped 1:1, a wide camera would put lights far outside the field the lens was designed for, where its rims clip
	// nearly every ghost away.
	constants.fieldScale = kSensorHalfHeight / focalLength_ / constants.tanHalfFovY;
	constants.ghostScale = std::clamp(flare.ghostSize, 0.01f, 1.0f);

	const ResourceHandle constantBuffer = constantBuffers_[renderContext_->GetCurrentFrameIndex()];
	renderContext_->UpdateConstantBuffer(constantBuffer, &constants, sizeof(constants));

	const rg::ResourceId results = frame.ImportBuffer(results_, "LensFlare_Results");
	const rg::ResourceId exposure = frame.ImportBuffer(exposure_, "AutoExposure_Exposure");
	const rg::ResourceId lights = frame.ImportBuffer(lights_, "LensFlare_Lights");
	const rg::ResourceId args = frame.ImportBuffer(args_, "LensFlare_Args");
	const rg::ResourceId vertices = frame.ImportBuffer(ghostVertices_, "LensFlare_GhostVertices");
	const rg::ResourceId apertures = frame.ImportBuffer(ghostApertures_, "LensFlare_GhostApertures");
	const rg::ResourceId quads = frame.ImportBuffer(quads_, "LensFlare_Quads");
	const rg::ResourceId output = bloom_.GetOutput();

	TextureDesc desc = {};
	desc.width = flareWidth_;
	desc.height = flareHeight_;
	desc.format = Format::R16G16B16A16_FLOAT;
	desc.debugName = "LensFlare_Target";
	const rg::ResourceId target = frame.graph.CreateTexture(desc);

	const rg::Access rw = rg::Access::ComputeShaderReadWrite;
	frame.AddPass(rg::Pass::Compute("LensFlare/Select")
						  .Use(results, rw)
						  .Use(exposure, rw)
						  .Use(lights, rw)
						  .Use(args, rw)
						  .Executes(
								  [this, constantBuffer, results, exposure, lights, args](rg::PassContext &ctx)
								  {
									  ctx.rc.SetComputePipeline(selectPipeline_);
									  ctx.rc.SetComputeConstantBufferByName("g_Constants", constantBuffer);
									  ctx.rc.SetComputeUAVByName("g_Results", ctx.Get(results));
									  ctx.rc.SetComputeUAVByName("g_Exposure", ctx.Get(exposure));
									  ctx.rc.SetComputeUAVByName("g_Lights", ctx.Get(lights));
									  ctx.rc.SetComputeUAVByName("g_Args", ctx.Get(args));
									  ctx.rc.Dispatch(1, 1, 1);
								  }));

	const uint32_t traceGroups = constants.maxLights * constants.ghostCount * FLARE_WAVELENGTHS;
	frame.AddPass(rg::Pass::Compute("LensFlare/Trace")
						  .Use(lights, rw)
						  .Use(args, rw)
						  .Use(vertices, rw)
						  .Use(apertures, rw)
						  .Use(quads, rw)
						  .Executes(
								  [this, constantBuffer, lights, args, vertices, apertures, quads, traceGroups](
										  rg::PassContext &ctx)
								  {
									  ctx.rc.SetComputePipeline(tracePipeline_);
									  ctx.rc.SetComputeConstantBufferByName("g_Constants", constantBuffer);
									  ctx.rc.SetComputeUAVByName("g_Lights", ctx.Get(lights));
									  ctx.rc.SetComputeUAVByName("g_Args", ctx.Get(args));
									  ctx.rc.SetComputeUAVByName("g_GhostVertices", ctx.Get(vertices));
									  ctx.rc.SetComputeUAVByName("g_GhostApertures", ctx.Get(apertures));
									  ctx.rc.SetComputeUAVByName("g_Quads", ctx.Get(quads));
									  ctx.rc.SetComputeSRVByName("g_Surfaces", surfaceBuffer_);
									  ctx.rc.SetComputeSRVByName("g_Ghosts", ghostBuffer_);
									  ctx.rc.Dispatch(traceGroups, 1, 1);
								  }));

	const rg::Access vsRead = rg::Access::VertexShaderRead;
	frame.AddPass(rg::Pass::Graphics("LensFlare/Draw")
						  .Writes(target)
						  .Use(lights, vsRead)
						  .Use(vertices, vsRead)
						  .Use(apertures, vsRead)
						  .Use(quads, vsRead)
						  .Use(args, rg::Access::IndirectArgumentRead)
						  .Executes(
								  [this, constantBuffer, target, lights, vertices, apertures, quads, args](
										  rg::PassContext &ctx)
								  {
									  RenderContext &rc = ctx.rc;
									  const Resource renderTarget = ctx.GetResource(target);
									  float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
									  rc.ClearRenderTarget(renderTarget, clear);
									  rc.SetRenderTargets(&renderTarget, 1);
									  rc.SetViewport(Viewport{ 0, 0, flareWidth_, flareHeight_, 0.0f, 1.0f });
									  rc.SetScissorRect(ScissorRect{ 0, 0, flareWidth_, flareHeight_ });

									  rc.SetGraphicsPipeline(ghostPipeline_);
									  rc.SetPrimitiveTopology(PrimitiveTopology::TriangleList);
									  rc.SetConstantBufferByName("g_Constants", constantBuffer);
									  rc.SetSRVByName("g_QuadList", ctx.Get(quads));
									  rc.SetSRVByName("g_Vertices", ctx.Get(vertices));
									  rc.SetSRVByName("g_Apertures", ctx.Get(apertures));
									  rc.ExecuteIndirect(drawSignature_, ctx.Get(args), 0, 1);

									  rc.SetGraphicsPipeline(starburstPipeline_);
									  rc.SetPrimitiveTopology(PrimitiveTopology::TriangleList);
									  rc.SetConstantBufferByName("g_Constants", constantBuffer);
									  rc.SetSRVByName("g_FlareLights", ctx.Get(lights));
									  rc.SetSRVByName("g_Starburst", starburst_);
									  rc.ExecuteIndirect(drawSignature_, ctx.Get(args), 4 * sizeof(uint32_t), 1);
								  }));

	frame.AddPass(rg::Pass::Compute("LensFlare/Composite")
						  .Reads(target)
						  .Use(output, rw)
						  .Executes(
								  [this, constantBuffer, target, output](rg::PassContext &ctx)
								  {
									  ctx.rc.SetComputePipeline(compositePipeline_);
									  ctx.rc.SetComputeConstantBufferByName("g_Constants", constantBuffer);
									  ctx.rc.SetComputeSRVByName("g_Flare", ctx.Get(target));
									  ctx.rc.SetComputeUAVByName("g_Output", ctx.Get(output));
									  ctx.rc.Dispatch((width_ + 7) / 8, (height_ + 7) / 8, 1);
								  }));
}
