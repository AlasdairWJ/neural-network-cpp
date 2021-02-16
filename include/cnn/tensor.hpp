#pragma once

#include <algorithm>
#include "shape.hpp"

// -----------------------------------------------------------------------------

template <typename Shape>
struct tensor;

template <unsigned Size>
using vector = tensor<shape_t<Size>>;

template <unsigned Size, typename Shape>
using vector_of = tensor<extend_shape_t<Size, Shape>>;

template <unsigned Rows, unsigned Cols>
using matrix = tensor<shape_t<Rows, Cols>>;

// -----------------------------------------------------------------------------

template <unsigned Size>
struct tensor<shape_t<Size>>
{
	static_assert(Size != 0);

	using shape = shape_t<Size>;
	using this_tensor = tensor<shape>;

	using raw_type = float[Size];

	float *begin() { return std::begin(values); }
	float *end() { return std::end(values); }

	const float *begin() const { return std::begin(values); }
	const float *end() const { return std::end(values); }

	constexpr float &at(const unsigned i) { return values[i]; }
	constexpr const float &at(const unsigned i) const { return values[i]; }

	constexpr float &operator[](const unsigned i) { return values[i]; }
	constexpr const float &operator[](const unsigned i) const { return values[i]; }

	this_tensor &set(const float value = 0.0f)
	{
		std::fill(std::begin(values), std::end(values), value);
		return *this;
	}

	this_tensor &zero()
	{
		return set(0.0);
	}

	this_tensor &operator=(const float value)
	{
		return set(value);
	}

	this_tensor &operator=(const raw_type &value)
	{
		*this = reinterpret_cast<const this_tensor &>(value);
	}

	this_tensor &operator+=(const this_tensor &other)
	{
		for (unsigned i = 0; i < Size; i++)
			values[i] += other.values[i];
		return *this;
	}

	this_tensor &operator-=(const this_tensor &other)
	{
		for (unsigned i = 0; i < Size; i++)
			values[i] -= other.values[i];
		return *this;
	}

	this_tensor &operator*=(const float factor)
	{
		for (unsigned i = 0; i < Size; i++)
			values[i] *= factor;
		return *this;
	}

	this_tensor &operator/=(const float factor)
	{
		for (unsigned i = 0; i < Size; i++)
			values[i] /= factor;
		return *this;
	}

	template <typename NewShape>
	auto &ravel()
	{
		static_assert(NewShape::count == shape::count, "can only ravel into shapes with the same number of elements");
		return *reinterpret_cast<tensor<NewShape>*>(this);
	}

	template <typename NewShape>
	const auto &ravel() const
	{
		static_assert(NewShape::count == shape::count, "can only ravel into shapes with the same number of elements");
		return *reinterpret_cast<const tensor<NewShape>*>(this);
	}

	template <unsigned Offset>
	auto &offset()
	{
		static_assert(Offset < Size);
		return *reinterpret_cast<vector<Size - Offset> *>(data() + Offset);
	}

	template <unsigned Offset>
	const auto &offset() const
	{
		static_assert(Offset < Size);
		return *reinterpret_cast<const vector<Size - Offset> *>(data() + Offset);
	}

	template <unsigned Length>
	auto &truncate()
	{
		static_assert(Length < Size);
		return *reinterpret_cast<vector<Length> *>(this);
	}

	template <unsigned Length>
	const auto &truncate() const
	{
		static_assert(Length < Size);
		return *reinterpret_cast<const vector<Length> *>(this);
	}

	float *data() { return reinterpret_cast<float *>(this); }
	const float *data() const { return reinterpret_cast<const float *>(this); }

private:
	float values[Size];
};

template <unsigned Size, unsigned NextSize, unsigned... RestSizes>
struct tensor<shape_t<Size, NextSize, RestSizes...>>
{
	static_assert(Size != 0);

	using shape = shape_t<Size, NextSize, RestSizes...>;
	using this_tensor = tensor<shape>;

	using next_shape = shape_t<NextSize, RestSizes...>;
	using next_tensor = tensor<next_shape>;

	using raw_type = typename next_tensor::raw_type[Size];

	next_tensor *begin() { return std::begin(values); }
	next_tensor *end() { return std::end(values); }

	const next_tensor *begin() const { return std::begin(values); }
	const next_tensor *end() const { return std::end(values); }

	constexpr next_tensor &at(const unsigned i) { return values[i]; }
	constexpr const next_tensor &at(const unsigned i) const { return values[i]; }

	constexpr next_tensor &operator[](const unsigned i) { return values[i]; }
	constexpr const next_tensor &operator[](const unsigned i) const { return values[i]; }

	this_tensor &set(const float value = 0.0f)
	{
		return unravel().set(value), *this;
	}

	this_tensor &zero()
	{
		return set(0.0);
	}

	this_tensor &operator=(const float value)
	{
		return set(value);
	}

	this_tensor &operator=(const raw_type &value)
	{
		*this = reinterpret_cast<const this_tensor &>(value);
	}

	this_tensor &operator+=(const this_tensor &other)
	{
		return unravel() += other, *this;
	}

	this_tensor &operator-=(const this_tensor &other)
	{
		return unravel() -= other, *this;
	}

	this_tensor &operator*=(const float factor)
	{
		return unravel() *= factor, *this;
	}

	this_tensor &operator/=(const float factor)
	{
		return unravel() /= factor, *this;
	}

	auto &unravel() { return *reinterpret_cast<vector<shape::count> *>(this); }
	const auto &unravel() const { return *reinterpret_cast<const vector<shape::count> *>(this); }

private:
	next_tensor values[Size];
};