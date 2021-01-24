#pragma once

#include <cmath>

#include "la.hpp"
#include "math.hpp"

// A selection of layer types and templates for layer types, and cost functions

namespace nn
{

// -----------------------------------------------------------------------------

namespace non_linearity_functions
{

// for use with layers::non_linearity

struct logistic
{
	static double evaluate(const double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}

	static double derivative(const double x, const double y)
	{
		return y * (1.0 - y);
	}
};

struct softplus
{
	static double evaluate(const double x)
	{
		return log(1.0 + exp(x));
	}

	static double derivative(const double x, const double y)
	{
		return 1.0 / (1.0 + exp(-x));
	}
};

} // non_linearity_functions

// -----------------------------------------------------------------------------

namespace pooling_methods
{

// for use with layers::pooling

struct max
{
	template <size_t PoolSize>
	double forward(const la::vector<PoolSize> &input)
	{
		return math::max(input);
	}

	template <size_t PoolSize>
	void backward(const la::vector<PoolSize> &input,
                  const double output,
                  la::vector<PoolSize> &delta_input,
                  const double delta_output)
	{
		for (size_t i = 0; i < PoolSize; i++)
			delta_input[i] = input[i] == output ? delta_output : 0.0;
	}
};

struct average
{
	template <size_t PoolSize>
	double forward(const la::vector<PoolSize> &input)
	{
		return math::average(input);
	}

	template <size_t PoolSize>
	void backward(const la::vector<PoolSize> &input,
                  const double output,
                  la::vector<PoolSize> &deltaInput,
                  const double delta_output)
	{
		for (size_t i = 0; i < PoolSize; i++)
			deltaInput[i] = delta_output / PoolSize;
	}
};

} // pooling_methods

// -----------------------------------------------------------------------------

namespace layers
{

template <typename FunctionType>
struct non_linearity
{
	template <size_t InputSize>
	struct type
	{
		static constexpr size_t OutputSize = InputSize;

		struct Params; // leave as incomplete type to imply no params

		static void forward(const la::vector<InputSize> &input,
                            la::vector<OutputSize> &output)
		{
			for (size_t i = 0; i < InputSize; i++)
				output[i] = FunctionType::evaluate(input[i]);
		}

		static void backward(const la::vector<InputSize> &input,
                             const la::vector<OutputSize> &output,
                             la::vector<InputSize> &delta_input,
                             const la::vector<OutputSize> &delta_output)
		{
			for (size_t i = 0; i < InputSize; i++)
				delta_input[i] = delta_output[i] * FunctionType::derivative(input[i], output[i]);
		}
	};
};

template <size_t InputSize>
using logistic = non_linearity<non_linearity_functions::logistic>::type<InputSize>;

template <size_t InputSize>
using softplus = non_linearity<non_linearity_functions::softplus>::type<InputSize>;

// -----------------------------------------------------------------------------

template <size_t InputSize>
struct softmax
{
	static constexpr size_t OutputSize = InputSize;

	struct Params;

	static void forward(const la::vector<InputSize> &input,
                        la::vector<OutputSize> &output)
	{
		const double maxValue = math::max(input);

		double sum = 0.0;
		for (size_t i = 0; i < InputSize; i++)
			sum += (output[i] = exp(input[i] - maxValue));

		for (size_t i = 0; i < InputSize; i++)
			output[i] /= sum;
	}

	static void backward(const la::vector<InputSize> &input,
                         const la::vector<OutputSize> &output,
                         la::vector<InputSize> &deltaInput,
                         const la::vector<OutputSize> &deltaOutput)
	{
		for (size_t i = 0; i < InputSize; i++)
		{
			deltaInput[i] = 0.0;
			for (size_t j = 0; j < OutputSize; j++)
				deltaInput[i] += deltaOutput[j] * (math::kdelta(i, j) - output[i]) * output[j];
		}
	}
};

// -----------------------------------------------------------------------------

template <size_t OutputSize_>
struct fully_connected
{
	template <size_t InputSize>
	struct type
	{
		static constexpr size_t OutputSize = OutputSize_;

		struct Params
		{
			using WeightType = la::matrix<InputSize, OutputSize>;
			using BiasType = la::vector<OutputSize>;

			WeightType weight;
			BiasType bias;

			static constexpr size_t Count = WeightType::Count + BiasType::Count;

			static void randomise(Params &params)
			{
				nn::util::randomise(params.weight.unravel()) /= OutputSize;
				nn::util::randomise(params.bias);
			}
		};

		static void forward(const la::vector<InputSize> &input,
                            la::vector<OutputSize> &output,
                            const Params &params)
		{
			la::product(output, input, params.weight) += params.bias;
		}

		static void backward(const la::vector<InputSize> &input,
                             const la::vector<OutputSize> &output,
                             const Params &params,
                             la::vector<InputSize> &delta_input,
                             const la::vector<OutputSize> &delta_output,
                             Params &delta_params)
		{
			for (size_t j = 0; j < OutputSize; j++)
			{
				// we're ADDING to the values that exist in deltaParams
				// bc allocating buffers for it would be a expensive.
				// it'll be zero'd and averaged outside of this function.
				delta_params.bias[j] += delta_output[j];
				for (size_t i = 0; i < InputSize; i++)
					delta_params.weight[i][j] += input[i] * delta_output[j];
			}

			la::product(delta_input, params.weight, delta_output);
		}
	};
};

template <size_t PoolSize, typename PoolMethod>
struct pooling
{
	template <size_t InputSize>
	struct type
	{
		static_assert(InputSize %PoolSize == 0);

		static constexpr size_t OutputSize = InputSize / PoolSize;

		struct Params;

		static void forward(const la::vector<InputSize> &input,
                            la::vector<OutputSize> &output)
		{
			const auto &input_m = input.ravel<OutputSize, PoolSize>();

			for (size_t j = 0; j < OutputSize; j++)
				output[j] = PoolMethod::forward(input_m[j]);

		}

		static void backward(const la::vector<InputSize> &input,
                             const la::vector<OutputSize> &output,
                             la::vector<InputSize> &delta_input,
                             const la::vector<OutputSize> &delta_output)
		{
			const auto &input_m = input.ravel<OutputSize, PoolSize>();
			auto &delta_input_m = delta_input.ravel<OutputSize, PoolSize>();

			for (size_t j = 0; j < OutputSize; j++)
				PoolMethod::backward(input_m[j], output[j], delta_input_m[j], delta_output[j]);
		}
	};
};

} // layers

} // nn