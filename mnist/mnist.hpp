#pragma once

#include <cstdint>
#include <cstdio>

// -----------------------------------------------------------------------------

template <typename T, unsigned Size>
bool _verify_idx_dimension(FILE* file)
{
	uint32_t size;
	fread(&size, sizeof(size), 1, file);
	if (_byteswap_ulong(size) != Size)
		return false;

	if constexpr (std::is_array_v<T>)
	{
		return _verify_idx_dimension<std::remove_extent_t<T>, std::extent_v<T>>(file);
	}

	return true;
}

template <typename T, unsigned Size>
bool load_idx(const char* filename, T (&data)[Size])
{
	FILE* file;

	if (0 != fopen_s(&file, filename, "rb"))
		return puts("failed to open"), false;

	uint32_t magic_number;
	fread(&magic_number, sizeof(magic_number), 1, file);

	if (!_verify_idx_dimension<T, Size>(file))
		return fclose(file), false;

	fread(&data, sizeof(data), 1, file);

	return fclose(file), true;
}