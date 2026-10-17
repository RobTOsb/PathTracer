#pragma once

#include "RenderContext.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <span>
#include <type_traits>
#include <vector>


// Bump when the layout or anything baked into the tables changes, so older files are rebuilt.
// 2: textures shared by materials are stored once.
// 3: textures store their format, BC7 where they could be compressed.
// 4: rebuilds the version 3 files, whose BC7 blocks the encoder wrote as zeros.
// 5: transmissive materials thin-walled (kThinWalledTransmission in RenderGPUScene.cpp).
// 6: thin-walled from EXT_materials_thin_wall only, diffuse transmission in GpuMaterial (128 bytes).
// 7: primitives without TEXCOORD_0 or indices load instead of being skipped.
// 8: KHR_materials_anisotropy in GpuMaterial (anisotropy, anisotropyTexture in place of the pads).
// 9: EXT_materials_subsurface, GpuMaterial 144 bytes (subsurface radius and anisotropy).
constexpr uint32_t kSceneCacheVersion = 9;

struct SceneCacheTexture
{
	uint64_t offset; // of mip 0 in the file
	uint32_t width;
	uint32_t height;
	uint32_t mipLevels;
	Format format; // R8G8B8A8_UNORM(_SRGB) or BC7_UNORM(_SRGB)
};

// The tightly packed rows of one mip (texels, or 4x4 blocks for BC7), with no pixels.
TextureMipData SceneCacheMipLayout(const SceneCacheTexture &texture, uint32_t mip);
uint64_t SceneCacheMipBytes(const SceneCacheTexture &texture, uint32_t mip);
uint64_t SceneCacheTextureBytes(const SceneCacheTexture &texture);

class SceneCacheWriter
{
public:
	SceneCacheWriter() = default;
	~SceneCacheWriter();

	SceneCacheWriter(const SceneCacheWriter &) = delete;
	SceneCacheWriter &operator=(const SceneCacheWriter &) = delete;

	bool Open(const std::filesystem::path &path);

	uint32_t AddTexture(uint32_t width, uint32_t height, Format format, std::span<const TextureMipData> mips);

	template<typename T>
	void Table(const std::vector<T> &table)
	{
		static_assert(std::is_trivially_copyable_v<T>);
		WriteTable(table.data(), table.size(), sizeof(T));
	}

	template<typename T>
	void Value(const T &value)
	{
		static_assert(std::is_trivially_copyable_v<T>);
		WriteTable(&value, 1, sizeof(T));
	}

	bool Finish();

	const std::filesystem::path &GetPath() const
	{
		return path_;
	}

private:
	void Write(const void *data, uint64_t size);
	void WriteTable(const void *data, uint64_t count, uint32_t elementSize);

	std::filesystem::path path_;
	std::filesystem::path tempPath_;
	FILE *file_ = nullptr;
	uint64_t position_ = 0;
	uint64_t tableOffset_ = 0; // 0 until the first table
	std::vector<SceneCacheTexture> textures_;
	bool failed_ = false;
};

class SceneCacheReader
{
public:
	SceneCacheReader() = default;
	~SceneCacheReader();

	SceneCacheReader(const SceneCacheReader &) = delete;
	SceneCacheReader &operator=(const SceneCacheReader &) = delete;

	bool Open(const std::filesystem::path &path);

	template<typename T>
	void Table(std::vector<T> &table)
	{
		static_assert(std::is_trivially_copyable_v<T>);
		uint64_t count = 0;
		if (!ReadTableHeader(sizeof(T), count))
		{
			table.clear();
			return;
		}
		table.resize(count);
		Read(table.data(), count * sizeof(T));
	}

	template<typename T>
	void Value(T &value)
	{
		static_assert(std::is_trivially_copyable_v<T>);
		uint64_t count = 0;
		if (ReadTableHeader(sizeof(T), count) && count == 1)
		{
			Read(&value, sizeof(T));
		}
		else
		{
			failed_ = true;
		}
	}

	// True when every table was read whole and the last ended where the directory begins.
	bool EndTables();

	const std::vector<SceneCacheTexture> &GetTextures() const
	{
		return textures_;
	}

	bool ReadTexture(uint32_t index, std::vector<uint8_t> &pixels);

private:
	bool ReadTableHeader(uint32_t elementSize, uint64_t &count);
	void Read(void *data, uint64_t size);

	FILE *file_ = nullptr;
	uint64_t position_ = 0;
	uint64_t directoryOffset_ = 0;
	std::vector<SceneCacheTexture> textures_;
	bool failed_ = false;
};
