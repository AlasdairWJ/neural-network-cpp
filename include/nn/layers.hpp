#pragma once 

#include <cmath>

#include "la.hpp"
#include "math.hpp"

// A selection of layer types and templates for layer types, and cost functions

namespace nn
{

// Layers with no parameters should define Params as this
struct NoParamsType
{
	static constexpr size_t Count = 0;
};

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

		using Params = NoParamsType;

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

	using Params = NoParamsType;

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
						 la::Vector<InputSize>& delta_input,
						 const la::Vector<OutputSize>& delta_output)
	{
		for (size_t i = 0; i < InputSize; i++)
		{
			delta_input[i] = 0.0;
			for (size_t j = 0; j < OutputSize; j++)
				delta_input[i] += delta_output[j] * (math::kdelta(i, j) - output[i]) * output[j];
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
				deltaParams.bias[j] = deltaOutput[j];
				for (size_t i = 0; i < InputSize; i++)
					deltaParams.weight[i][j] = input[i] * deltaOutput[j];
			}

			la::product(deltaInput, params.weight, deltaOutput);
		}
	};
};

} // nn