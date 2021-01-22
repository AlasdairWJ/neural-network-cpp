#pragma once

#include "la.hpp"

namespace nn
{

namespace math
{

double kdelta(const size_t i, const size_t j)
{
	return i == j ? 1.0 : 0.0;
}

template <size_t N>
size_t argmax(const la::Vector<N> &x)
{
	size_t best_n = 0;
	for (size_t n = 1; n < N; n++)
		if (x[n] > x[best_n])
			best_n = n;
	return best_n;
}

template <size_t N>
double max(const la::Vector<N> &x)
{
	return x[argmax(x)];
}

template <size_t N>
size_t argmin(const la::Vector<N> &x)
{
	size_t best_n = 0;
	for (size_t n = 1; n < N; n++)
		if (x[n] < x[best_n])
			best_n = n;
	return best_n;
}

template <size_t N>
double min(const la::Vector<N> &x)
{
	return x[argmin(x)];
}

} // math

}