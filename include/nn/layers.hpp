#pragma once

#include <cmath>

#include "la.hpp"
#include "math.hpp"

// A selection of layer types and templates for layer types, and cost functions

namespace nn
{

// -----------------------------------------------------------------------------

// For use with NonLinearityLayer
struct LogisticFunction
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

// For use with NonLinearityLayer
struct SoftplusFunction
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

// -----------------------------------------------------------------------------

template <typename FunctionType>
struct NonLinearityLayer
{
	template <size_t InputSize>
	struct Type
	{
		static constexpr size_t OutputSize = InputSize;

		struct Params; // leave as incomplete type to imply no params

		static void forward(const la::Vector<InputSize>& input,
							la::Vector<OutputSize>& output)
		{
			for (size_t i = 0; i < InputSize; i++)
				output[i] = FunctionType::evaluate(input[i]);
		}

		static void backward(const la::Vector<InputSize>& input,
							 const la::Vector<OutputSize>& output,
							 la::Vector<InputSize>& deltaInput,
							 const la::Vector<OutputSize>& deltaOutput)
		{
			for (size_t i = 0; i < InputSize; i++)
				deltaInput[i] = deltaOutput[i] * FunctionType::derivative(input[i], output[i]);
		}
	};
};

// -----------------------------------------------------------------------------

template <size_t InputSize>
struct SoftmaxLayer
{
	static constexpr size_t OutputSize = InputSize;

	struct Params;

	static void forward(const la::Vector<InputSize>& input,
						la::Vector<OutputSize>& output)
	{
		const double maxValue = math::max(input);

		double sum = 0.0;
		for (size_t i = 0; i < InputSize; i++)
			sum += (output[i] = exp(input[i] - maxValue));

		for (size_t i = 0; i < InputSize; i++)
			output[i] /= sum;
	}

	static void backward(const la::Vector<InputSize>& input,
						 const la::Vector<OutputSize>& output,
						 la::Vector<InputSize>& deltaInput,
						 const la::Vector<OutputSize>& deltaOutput)
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
struct FullyConnectedLayer
{
	template <size_t InputSize>
	struct Type
	{
		static constexpr size_t OutputSize = OutputSize_;

		struct Params
		{
			using WeightType = la::Matrix<InputSize, OutputSize>;
			using BiasType = la::Vector<OutputSize>;

			WeightType weight;
			BiasType bias;

			static constexpr size_t Count = WeightType::Count + BiasType::Count;
		};

		static void forward(const la::Vector<InputSize>& input,
							la::Vector<OutputSize>& output,
							const Params& params)
		{
			la::product(output, input, params.weight) += params.bias;
		}

		static void backward(const la::Vector<InputSize>& input,
							 const la::Vector<OutputSize>& output,
							 const Params& params,
							 la::Vector<InputSize>& deltaInput,
							 const la::Vector<OutputSize>& deltaOutput,
							 Params& deltaParams)
		{
			for (size_t j = 0; j < OutputSize; j++)
			{
				// we're ADDING to the values that exist in deltaParams
				// bc allocating buffers for it would be a expensive.
				// it'll be zero'd and averaged outside of this function.
				deltaParams.bias[j] += deltaOutput[j];
				for (size_t i = 0; i < InputSize; i++)
					deltaParams.weight[i][j] += input[i] * deltaOutput[j];
			}

			la::product(deltaInput, params.weight, deltaOutput);
		}
	};
};

// -----------------------------------------------------------------------------

// One dimensional pooling, but still functional

struct MaxPool
{
	template <size_t PoolSize>
	double forward(const la::Vector<PoolSize>& input)
	{
		return math::max(input);
	}

	template <size_t PoolSize>
	void backward(const la::Vector<PoolSize>& input,
					const double output,
					la::Vector<PoolSize>& deltaInput,
					const double deltaOutput)
	{
		for (size_t i = 0; i < PoolSize; i++)
			deltaInput[i] = input[i] == output ? deltaOutput : 0.0;
	}
};

struct AveragePool
{
	template <size_t PoolSize>
	double forward(const la::Vector<PoolSize>& input)
	{
		return math::average(input);
	}

	template <size_t PoolSize>
	void backward(const la::Vector<PoolSize>& input,
					const double output,
					la::Vector<PoolSize>& deltaInput,
					const double deltaOutput)
	{
		for (size_t i = 0; i < PoolSize; i++)
			deltaInput[i] = deltaOutput / PoolSize;
	}
};

template <size_t PoolSize, typename PoolMethod>
struct PoolingLayer
{
	template <size_t InputSize>
	struct Type
	{
		static_assert(InputSize % PoolSize == 0);

		static constexpr size_t OutputSize = InputSize / PoolSize;

		struct Params;

		static void forward(const la::Vector<InputSize>& input,
							la::Vector<OutputSize>& output)
		{
			const auto& inputMatrix = input.ravel<OutputSize, PoolSize>();

			for (size_t j = 0; j < OutputSize; j++)
				output[j] = PoolMethod::forward(inputMatrix[j]);

		}

		static void backward(const la::Vector<InputSize>& input,
							 const la::Vector<OutputSize>& output,
							 la::Vector<InputSize>& deltaInput,
							 const la::Vector<OutputSize>& deltaOutput)
		{
			const auto& inputMatrix = input.ravel<OutputSize, PoolSize>();
			auto& deltaInputMatrix = input.ravel<OutputSize, PoolSize>();

			for (size_t j = 0; j < OutputSize; j++)
				PoolMethod::backward(inputMatrix[j], output[j], deltaInputMatrix[j], deltaOutput[j]);
		}
	};
};

} // nn