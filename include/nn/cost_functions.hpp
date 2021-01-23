#pragma once

#include <cmath>

namespace nn
{

struct CrossEntropyCostFunction
{
	template <size_t N>
	static double cost(const la::Vector<N>& expectation,
					   const la::Vector<N>& prediction)
	{
		double value = 0.0;
		for (size_t i = 0; i < N; i++)
			value -= expectation[i] * log(prediction[i]) + (1.0 - expectation[i]) * log(1.0 - prediction[i]);
		return value;
	}

	template <size_t N>
	static void derivative(const la::Vector<N>& expectation,
						   const la::Vector<N>& prediction,
						   la::Vector<N>& delta)
	{
		for (size_t i = 0; i < N; i++)
			delta[i] = (prediction[i] - expectation[i]) / (prediction[i] * (1.0 - prediction[i]));
	}
};

// -----------------------------------------------------------------------------

struct SumOfSquaredErrorsCostFunction
{
	template <size_t N>
	static double cost(const la::Vector<N>& expectation,
					   const la::Vector<N>& prediction)
	{
		double value = 0.0;
		for (size_t i = 0; i < N; i++)
			value += (prediction[i] - expectation[i]) * (prediction[i] - expectation[i]);
		return value / 2;
	}

	template <size_t N>
	static void derivative(const la::Vector<N>& expectation,
						   const la::Vector<N>& prediction,
						   la::Vector<N>& delta)
	{
		for (size_t i = 0; i < N; i++)
			delta[i] = prediction[i] - expectation[i];
	}
};
}