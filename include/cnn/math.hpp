#pragma once

#include "tensor.hpp"

namespace nn
{

namespace math
{

float kdelta(unsigned i, unsigned j)
{
	return i == j ? 1.0f : 0.0f;
}

template <unsigned N>
unsigned argmax(const vector<N> &values)
{
	unsigned max_i = 0;
	for (unsigned i = 1; i < N; i++)
		if (values[i] > values[max_i])
			max_i = i;
	return max_i;
}

template <unsigned N>
float max(const vector<N> &values)
{
	return values[argmax(values)];
}

template <unsigned N>
unsigned argmin(const vector<N> &values)
{
	unsigned min_i = 0;
	for (unsigned i = 1; i < N; i++)
		if (values[i] < values[min_i])
			min_i = i;
	return min_i;
}

template <unsigned N>
float min(const vector<N> &values)
{
	return values[argmin(values)];
}

template <unsigned N>
float dot(const vector<N> &a, const vector<N> &b)
{
	float v = 0.0;
	for (unsigned n = 0; n < N; n++)
		v += a[n] * b[n];
	return v;
}

template <unsigned N, unsigned M>
auto product(vector<M> &result, const vector<N> &lhs, const matrix<N, M> &rhs) -> decltype(result)
{
	for (unsigned m = 0; m < M; m++)
	{
		float &v = result[m] = 0.0f;
		for (unsigned n = 0; n < N; n++)
			v += lhs[n] * rhs[n][m];
	}
	return result;
}

template <unsigned N, unsigned M>
auto product(vector<N> &result, const matrix<N, M> &lhs, const vector<M> &rhs) -> decltype(result)
{
	for (unsigned n = 0; n < N; n++)
	{
		float &v = result[n] = 0.0f;
		for (unsigned m = 0; m < M; m++)
			v += lhs[n][m] * rhs[m];
	}
	return result;
}

template <int I, int J, int K>
auto product(matrix<I, J> &result, const matrix<I, K> &lhs, const matrix<K, J> &rhs) -> decltype(result)
{
	for (unsigned i = 0; i < I; i++)
		for (unsigned j = 0; j < J; j++)
		{
			float &v = result[i][j] = 0.0f;
			for (unsigned k = 0; k < K; k++)
				v += lhs[i][k] * rhs[k][j];
		}
	return result;
}

} // namespace math

} // namespace nn