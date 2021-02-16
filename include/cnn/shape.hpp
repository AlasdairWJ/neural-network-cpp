#pragma once

template <unsigned... Sizes>
struct shape_t;

template <unsigned Size>
struct shape_t<Size>
{
	static constexpr unsigned count = Size;
	static constexpr unsigned dim = 1;
};

template <unsigned Size, unsigned NextSize, unsigned... RestSizes>
struct shape_t<Size, NextSize, RestSizes...>
{
	using next_shape = shape_t<NextSize, RestSizes...>;

	static constexpr unsigned count = Size * next_shape::count;
	static constexpr unsigned dim = 1 + next_shape::dim;
};

// -----------------------------------------------------------------------------

template <unsigned Size, typename Shape>
struct extend_shape;

template <unsigned Size, unsigned... Sizes>
struct extend_shape<Size, shape_t<Sizes...>>
{
	using type = shape_t<Size, Sizes...>;
};

template <unsigned Size, typename Shape>
using extend_shape_t = typename extend_shape<Size, Shape>::type;
