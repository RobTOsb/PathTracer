#include "SceneCache.h"

#include "fmtlog.h"

#include <algorithm>
#include <cstring>

namespace
{
	struct SceneCacheHeader
	{
		char magic[8];
		uint32_t version;
		uint32_t textureCount;
		uint64_t tableOffset;
		uint64_t directoryOffset; // the directory runs to the end of the file
		uint64_t fileSize;
	};

	constexpr char kMagic[8] = "PTSCENE";

	FILE *OpenFile(const std::filesystem::path &path, const wchar_t *mode)
	{
		FILE *file = nullptr;
		return _wfopen_s(&file, path.c_str(), mode) == 0 ? file : nullptr;
	}

	// Precedes each table.
	struct TableHeader
	{
		uint64_t count;
		uint32_t elementSize;
		uint32_t padding;
	};
} // namespace

TextureMipData SceneCacheMipLayout(const SceneCacheTexture &texture, uint32_t mip)
{
	const uint32_t width = std::max(texture.width >> mip, 1u);
	const uint32_t height = std::max(texture.height >> mip, 1u);
	TextureMipData layout = {};
	if (texture.format == Format::BC7_UNORM || texture.format == Format::BC7_UNORM_SRGB)
	{
		layout.rowPitch = (width + 3) / 4 * 16;
		layout.rowCount = (height + 3) / 4;
	}
	else
	{
		layout.rowPitch = width * 4;
		layout.rowCount = height;
	}
	return layout;
}

uint64_t SceneCacheMipBytes(const SceneCacheTexture &texture, uint32_t mip)
{
	const TextureMipData layout = SceneCacheMipLayout(texture, mip);
	return static_cast<uint64_t>(layout.rowPitch) * layout.rowCount;
}

uint64_t SceneCacheTextureBytes(const SceneCacheTexture &texture)
{
	uint64_t bytes = 0;
	for (uint32_t m = 0; m < texture.mipLevels; ++m)
	{
		bytes += SceneCacheMipBytes(texture, m);
	}
	return bytes;
}

SceneCacheWriter::~SceneCacheWriter()
{
	if (file_ != nullptr)
	{
		fclose(file_);
		std::error_code error;
		std::filesystem::remove(tempPath_, error);
	}
}

bool SceneCacheWriter::Open(const std::filesystem::path &path)
{
	path_ = path;
	tempPath_ = path;
	tempPath_ += ".tmp";
	file_ = OpenFile(tempPath_, L"wb");
	if (file_ == nullptr)
	{
		logw("SceneCache: could not create '{}'", tempPath_.string());
		return false;
	}

	// Rewritten by Finish once the offsets are known.
	const SceneCacheHeader header = {};
	Write(&header, sizeof(header));
	return !failed_;
}

void SceneCacheWriter::Write(const void *data, uint64_t size)
{
	if (failed_ || size == 0)
	{
		return;
	}
	if (fwrite(data, 1, size, file_) != size)
	{
		failed_ = true;
	}
	position_ += size;
}

uint32_t SceneCacheWriter::AddTexture(uint32_t width, uint32_t height, Format format, std::span<const TextureMipData> mips)
{
	SceneCacheTexture texture = {};
	texture.offset = position_;
	texture.width = width;
	texture.height = height;
	texture.mipLevels = static_cast<uint32_t>(mips.size());
	texture.format = format;

	for (uint32_t m = 0; m < mips.size(); ++m)
	{
		const uint64_t bytes = static_cast<uint64_t>(mips[m].rowPitch) * mips[m].rowCount;
		if (tableOffset_ != 0 || bytes != SceneCacheMipBytes(texture, m))
		{
			failed_ = true; // after the tables, or not tightly packed
		}
		Write(mips[m].pixels, bytes);
	}

	textures_.push_back(texture);
	return static_cast<uint32_t>(textures_.size() - 1);
}

void SceneCacheWriter::WriteTable(const void *data, uint64_t count, uint32_t elementSize)
{
	if (tableOffset_ == 0)
	{
		tableOffset_ = position_;
	}
	const TableHeader header = { count, elementSize, 0 };
	Write(&header, sizeof(header));
	Write(data, count * elementSize);
}

bool SceneCacheWriter::Finish()
{
	if (tableOffset_ == 0)
	{
		tableOffset_ = position_;
	}

	SceneCacheHeader header = {};
	std::memcpy(header.magic, kMagic, sizeof(kMagic));
	header.version = kSceneCacheVersion;
	header.textureCount = static_cast<uint32_t>(textures_.size());
	header.tableOffset = tableOffset_;
	header.directoryOffset = position_;
	Write(textures_.data(), textures_.size() * sizeof(SceneCacheTexture));
	header.fileSize = position_;

	if (!failed_ && _fseeki64(file_, 0, SEEK_SET) == 0)
	{
		Write(&header, sizeof(header));
	}
	else
	{
		failed_ = true;
	}

	const bool closed = fclose(file_) == 0;
	file_ = nullptr;

	std::error_code error;
	if (failed_ || !closed)
	{
		logw("SceneCache: failed to write '{}'", tempPath_.string());
		std::filesystem::remove(tempPath_, error);
		return false;
	}
	std::filesystem::rename(tempPath_, path_, error);
	if (error)
	{
		logw("SceneCache: could not replace '{}': {}", path_.string(), error.message());
		std::filesystem::remove(tempPath_, error);
		return false;
	}
	return true;
}

SceneCacheReader::~SceneCacheReader()
{
	if (file_ != nullptr)
	{
		fclose(file_);
	}
}

bool SceneCacheReader::Open(const std::filesystem::path &path)
{
	std::error_code error;
	const uint64_t fileSize = std::filesystem::file_size(path, error);
	if (error)
	{
		return false;
	}

	file_ = OpenFile(path, L"rb");
	if (file_ == nullptr)
	{
		logw("SceneCache: could not open '{}'", path.string());
		return false;
	}

	SceneCacheHeader header = {};
	Read(&header, sizeof(header));
	if (failed_ || std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0)
	{
		logw("SceneCache: '{}' is not a scene cache", path.string());
		return false;
	}
	if (header.version != kSceneCacheVersion)
	{
		logi("SceneCache: '{}' is version {}, expected {}", path.string(), header.version, kSceneCacheVersion);
		return false;
	}
	if (header.fileSize != fileSize || header.tableOffset < sizeof(header) ||
		header.directoryOffset < header.tableOffset ||
		fileSize - header.directoryOffset != static_cast<uint64_t>(header.textureCount) * sizeof(SceneCacheTexture))
	{
		logw("SceneCache: '{}' is truncated or damaged", path.string());
		return false;
	}

	textures_.resize(header.textureCount);
	if (_fseeki64(file_, static_cast<int64_t>(header.directoryOffset), SEEK_SET) != 0)
	{
		return false;
	}
	position_ = header.directoryOffset;
	Read(textures_.data(), textures_.size() * sizeof(SceneCacheTexture));

	// The pixels must sit between the header and the tables.
	for (const SceneCacheTexture &texture: textures_)
	{
		const bool bc7 = texture.format == Format::BC7_UNORM || texture.format == Format::BC7_UNORM_SRGB;
		const bool validFormat = bc7 ? texture.width % 4 == 0 && texture.height % 4 == 0
									 : texture.format == Format::R8G8B8A8_UNORM ||
											   texture.format == Format::R8G8B8A8_UNORM_SRGB;
		const bool valid = validFormat && texture.width > 0 && texture.height > 0 && texture.width <= 16384 &&
						   texture.height <= 16384 && texture.mipLevels > 0 && texture.mipLevels <= 15 &&
						   texture.offset >= sizeof(header) && texture.offset <= header.tableOffset &&
						   SceneCacheTextureBytes(texture) <= header.tableOffset - texture.offset;
		if (!valid)
		{
			failed_ = true;
		}
	}

	if (failed_ || _fseeki64(file_, static_cast<int64_t>(header.tableOffset), SEEK_SET) != 0)
	{
		logw("SceneCache: '{}' is damaged", path.string());
		return false;
	}
	position_ = header.tableOffset;
	directoryOffset_ = header.directoryOffset;
	return true;
}

void SceneCacheReader::Read(void *data, uint64_t size)
{
	if (failed_ || size == 0)
	{
		return;
	}
	if (fread(data, 1, size, file_) != size)
	{
		failed_ = true;
	}
	position_ += size;
}

bool SceneCacheReader::ReadTableHeader(uint32_t elementSize, uint64_t &count)
{
	TableHeader header = {};
	Read(&header, sizeof(header));
	// A size mismatch means a struct changed without a version bump.
	if (failed_ || header.elementSize != elementSize || position_ > directoryOffset_ ||
		header.count > (directoryOffset_ - position_) / elementSize)
	{
		failed_ = true;
		return false;
	}
	count = header.count;
	return true;
}

bool SceneCacheReader::EndTables()
{
	return !failed_ && position_ == directoryOffset_;
}

bool SceneCacheReader::ReadTexture(uint32_t index, std::vector<uint8_t> &pixels)
{
	const SceneCacheTexture &texture = textures_[index];
	pixels.resize(SceneCacheTextureBytes(texture));
	return _fseeki64(file_, static_cast<int64_t>(texture.offset), SEEK_SET) == 0 &&
		   fread(pixels.data(), 1, pixels.size(), file_) == pixels.size();
}
