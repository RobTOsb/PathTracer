#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Film look LUTs: 3D DDS volumes (R16G16B16A16_FLOAT, N^3, red along x) in assets/luts, made from .cube files by
// tools/cube_to_dds.py. Each maps display-encoded (sRGB) colour to display-encoded colour.
struct FilmLutFile
{
	std::string name; // for the UI: the file name without extension, underscores as spaces
	std::filesystem::path path;
};

// The LUTs in assets/luts, sorted by name; scanned on first use. RenderSettings::filmLut 0 is none, i is
// GetFilmLutFiles()[i - 1].
const std::vector<FilmLutFile> &GetFilmLutFiles();

struct FilmLutData
{
	uint32_t size = 0; // texels per side
	std::vector<uint16_t> texels; // size^3 RGBA halves
};

// Throws std::runtime_error when the file is not such a DDS.
FilmLutData LoadFilmLut(const std::filesystem::path &path);
