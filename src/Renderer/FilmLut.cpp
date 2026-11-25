#include "FilmLut.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace
{
	constexpr uint32_t kDxgiFormatR16G16B16A16Float = 10;
	constexpr uint32_t kResourceDimensionTexture3D = 4;
	constexpr uint32_t kHeaderSize = 4 + 124 + 20; // magic, DDS_HEADER, DDS_HEADER_DXT10
	constexpr uint32_t kMaxLutSize = 256;

	uint32_t ReadU32(const std::vector<char> &bytes, size_t offset)
	{
		uint32_t value;
		std::memcpy(&value, bytes.data() + offset, sizeof(value));
		return value;
	}
} // namespace

const std::vector<FilmLutFile> &GetFilmLutFiles()
{
	static const std::vector<FilmLutFile> files = []
	{
		std::vector<FilmLutFile> result;
		const std::filesystem::path directory = std::filesystem::path(PATHTRACER_ASSET_DIR) / "luts";
		std::error_code error;
		for (const auto &entry: std::filesystem::directory_iterator(directory, error))
		{
			std::string extension = entry.path().extension().string();
			std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
			if (!entry.is_regular_file() || extension != ".dds")
			{
				continue;
			}
			std::string name = entry.path().stem().string();
			std::replace(name.begin(), name.end(), '_', ' ');
			result.push_back({ name, entry.path() });
		}
		std::sort(result.begin(),
				  result.end(),
				  [](const FilmLutFile &a, const FilmLutFile &b) { return a.name < b.name; });
		return result;
	}();
	return files;
}

FilmLutData LoadFilmLut(const std::filesystem::path &path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file)
	{
		throw std::runtime_error("Film LUT: cannot open " + path.string());
	}
	const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (bytes.size() < kHeaderSize || std::memcmp(bytes.data(), "DDS ", 4) != 0 || std::memcmp(bytes.data() + 84, "DX10", 4) != 0)
	{
		throw std::runtime_error("Film LUT: " + path.string() + " is not a DDS with a DX10 header");
	}
	const uint32_t height = ReadU32(bytes, 12);
	const uint32_t width = ReadU32(bytes, 16);
	const uint32_t depth = ReadU32(bytes, 24);
	const uint32_t format = ReadU32(bytes, 128);
	const uint32_t dimension = ReadU32(bytes, 132);
	if (format != kDxgiFormatR16G16B16A16Float || dimension != kResourceDimensionTexture3D)
	{
		throw std::runtime_error("Film LUT: " + path.string() + " is not an R16G16B16A16_FLOAT volume");
	}
	if (width != height || width != depth || width < 2 || width > kMaxLutSize)
	{
		throw std::runtime_error("Film LUT: " + path.string() + " is not a cube of 2 to 256 texels per side");
	}
	const size_t texelCount = static_cast<size_t>(width) * width * width;
	if (bytes.size() < kHeaderSize + texelCount * 8)
	{
		throw std::runtime_error("Film LUT: " + path.string() + " is truncated");
	}
	FilmLutData lut;
	lut.size = width;
	lut.texels.resize(texelCount * 4);
	std::memcpy(lut.texels.data(), bytes.data() + kHeaderSize, texelCount * 8);
	return lut;
}
