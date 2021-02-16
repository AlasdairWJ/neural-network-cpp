#pragma once

#include <cmath>

namespace nn
{

namespace cost_functions
{

struct cross_entropy
{
	template <unsigned N>
	static float cost(const vector<N> &expectation,
                       const vector<N> &prediction)
	{
		float value = 0.0f;
		for (unsigned i = 0; i < N; i++)
			value -= expectation[i] * log(prediction[i]) + (1.0f - expectation[i]) * log(1.0f - prediction[i]);
		return value;
	}

	template <unsigned N>
	static void derivative(const vector<N> &expectation,
                           const vector<N> &prediction,
                           vector<N> &delta)
	{
		for (unsigned i = 0; i < N; i++)
			delta[i] = (prediction[i] - expectation[i]) / (prediction[i] * (1.0f - prediction[i]));
	}
};

// -----------------------------------------------------------------------------

struct sum_of_squared_errors
{
	template <unsigned N>
	static float cost(const vector<N> &expectation,
                       const vector<N> &prediction)
	{
		float value = 0.0f;
		for (unsigned i = 0; i < N; i++)
			value += (prediction[i] - expectation[i]) * (prediction[i] - expectation[i]);
		return 0.5f * value;
	}

	template <unsigned N>
	static void derivative(const vector<N> &expectation,
                           const vector<N> &prediction,
                           vector<N> &delta)
	{
		for (unsigned i = 0; i < N; i++)
			delta[i] = prediction[i] - expectation[i];
	}
};

} // cost_functions

} // nn