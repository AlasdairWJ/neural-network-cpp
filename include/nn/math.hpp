#pragma once

#include "la.hpp"

namespace nn
{

namespace math
{

template <typename T>
double kdelta(const T i, const T j)
{
	return i == j ? 1.0 : 0.0; // keep things type-flexible
}

template <size_t N>
size_t argmax(const la::vector<N> &x)
{
	size_t best_n = 0;
	for (size_t n = 1; n < N; n++)
		if (x[n] > x[best_n])
			best_n = n;
	return best_n;
}

template <size_t N>
double max(const la::vector<N> &x)
{
	return x[argmax(x)];
}

template <size_t N>
size_t argmin(const la::vector<N> &x)
{
	size_t best_n = 0;
	for (size_t n = 1; n < N; n++)
		if (x[n] < x[best_n])
			best_n = n;
	return best_n;
}

template <size_t N>
double min(const la::vector<N> &x)
{
	return x[argmin(x)];
}

template <size_t N>
double sum(const la::vector<N> &x)
{
	double total = 0;
	for (const double& value : x)
		total += value;
	return total;
}

template <size_t N>
double average(const la::vector<N> &x)
{
	return sum(x) / N;
}

} // math

}