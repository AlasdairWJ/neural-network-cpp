#pragma once

#include <cmath>
#include <type_traits>

#include "tensor.hpp"
#include "math.hpp"

// A selection of layer types and templates for layer types

namespace nn
{

// -----------------------------------------------------------------------------

namespace non_linearity_functions
{

// for use with layers::non_linearity

struct logistic
{
	static float evaluate(const float x)
	{
		return 1.0f / (1.0f + exp(-x));
	}

	static float derivative(const float x, const float y)
	{
		return y * (1.0f - y);
	}
};

struct relu
{
	static float evaluate(const float x)
	{
		return std::max(0.0f, x);
	}

	static float derivative(const float x, const float y)
	{
		return x > 0.0f ? 1.0f : 0.0f;
	}
};

struct softplus
{
	static float evaluate(const float x)
	{
		return log(1.0f + exp(x));
	}

	static float derivative(const float x, const float y)
	{
		return 1.0f / (1.0f + exp(-x));
	}
};

} // non_linearity_functions

// -----------------------------------------------------------------------------

namespace layers
{

template <typename FunctionType>
struct non_linearity
{
	template <typename InputShape>
	struct type
	{
		using output_shape = InputShape;

		struct params_t; // leave as incomplete type to imply no params

		static void forward(const vector<InputShape::count> &input,
                            vector<InputShape::count> &output)
		{
			for (unsigned i = 0; i < InputShape::count; i++)
				output[i] = FunctionType::evaluate(input[i]);
		}

		template <typename = std::enable_if_t<InputShape::dim != 1>>
		static void forward(const tensor<InputShape> &input,
                            tensor<InputShape> &output)
		{
			forward(input.unravel(), output.unravel());
		}

		static void backward(const vector<InputShape::count> &input,
                             const vector<output_shape::count> &output,
                             vector<InputShape::count> &delta_input,
                             const vector<output_shape::count> &delta_output)
		{
			for (unsigned i = 0; i < InputShape::count; i++)
				delta_input[i] = delta_output[i] * FunctionType::derivative(input[i], output[i]);
		}

		template <typename = std::enable_if_t<InputShape::dim != 1>>
		static void backward(const tensor<InputShape> &input,
                             const tensor<output_shape> &output,
                             tensor<InputShape> &delta_input,
                             const tensor<output_shape> &delta_output)
		{
			backward(input.unravel(), output.unravel(), delta_input.unravel(), delta_output.unravel());
		}
	};
};

template <typename InputShape>
using logistic = non_linearity<non_linearity_functions::logistic>::type<InputShape>;

template <typename InputShape>
using softplus = non_linearity<non_linearity_functions::softplus>::type<InputShape>;

template <typename InputShape>
using relu = non_linearity<non_linearity_functions::relu>::type<InputShape>;

// -----------------------------------------------------------------------------

template <typename InputShape>
struct softmax
{
	using output_shape = shape_t<InputShape::count>;

	struct params_t;

	static void forward(const vector<InputShape::count> &input,
                        tensor<output_shape> &output)
	{
		const float maxValue = math::max(input);

		float sum = 0.0;
		for (unsigned i = 0; i < InputShape::count; i++)
			sum += (output[i] = exp(input[i] - maxValue));

		for (unsigned i = 0; i < InputShape::count; i++)
			output[i] /= sum;
	}

	template <typename = std::enable_if_t<InputShape::dim != 1>>
	static void forward(const tensor<InputShape> &input,
                        tensor<output_shape> &output)
	{
		forward(input.unravel(), output);
	}

	static void backward(const vector<InputShape::count> &input,
                         const tensor<output_shape> &output,
                         vector<InputShape::count> &delta_input,
                         const tensor<output_shape> &delta_output)
	{
		for (unsigned i = 0; i < InputShape::count; i++)
		{
			delta_input[i] = 0.0;
			for (unsigned j = 0; j < InputShape::count; j++)
				delta_input[i] += delta_output[j] * (math::kdelta(i, j) - output[i]) * output[j];
		}
	}

	template <typename = std::enable_if_t<InputShape::dim != 1>>
	static void backward(const tensor<InputShape> &input,
                         const tensor<output_shape> &output,
                         tensor<InputShape> &delta_input,
                         const tensor<output_shape> &delta_output)
	{
		backward(input.unravel(), output, delta_input.unravel(), delta_output);
	}
};

// -----------------------------------------------------------------------------

template <unsigned OutputSize>
struct fully_connected
{
	template <typename InputShape>
	struct type
	{
		using output_shape = shape_t<OutputSize>;

		struct params_t
		{
			matrix<InputShape::count, OutputSize> weight;
			vector<OutputSize> bias;

			void randomise()
			{
				nn::util::randomise(weight) /= InputShape::count;
				nn::util::randomise(bias);
			}
		};

		static void forward(const vector<InputShape::count> &input,
                            vector<OutputSize> &output,
                            const params_t &params)
		{
			math::product(output, input, params.weight);
			output += params.bias;
		}

		template <typename = std::enable_if_t<InputShape::dim != 1>>
		static void forward(const tensor<InputShape> &input,
                            vector<OutputSize> &output,
                            const params_t &params)
		{
			forward(input.unravel(), output, params);
		}

		static void backward(const vector<InputShape::count> &input,
                             const vector<OutputSize> &output,
                             const params_t &params,
                             vector<InputShape::count> &delta_input,
                             const vector<OutputSize> &delta_output,
                             params_t &delta_params)
		{
			for (unsigned j = 0; j < OutputSize; j++)
			{
				// we're ADDING to the values that exist in deltaParams
				// bc allocating buffers for it would be a expensive.
				// it'll be zero'd and averaged outside of this function.
				delta_params.bias[j] += delta_output[j];
				for (unsigned i = 0; i < InputShape::count; i++)
					delta_params.weight[i][j] += input[i] * delta_output[j];
			}

			math::product(delta_input, params.weight, delta_output);
		}

		template <typename = std::enable_if_t<InputShape::dim != 1>>
		static void backward(const tensor<InputShape> &input,
                             const vector<OutputSize> &output,
                             const params_t &params,
                             tensor<InputShape> &delta_input,
                             const vector<OutputSize> &delta_output,
                             params_t &delta_params)
		{
			backward(input.unravel(), output, params,
                     delta_input.unravel(), delta_output, delta_params);
		}
	};
};

namespace pooling_methods
{

// for use with layers::pooling

struct max
{
	template <unsigned PoolSize, unsigned InputRows, unsigned InputCols>
	float forward(const matrix<InputRows, InputCols> &input, const unsigned oi, const unsigned oj)
	{
		float value = input[0][0];
		for (unsigned i = 0; i < PoolSize; i++)
			for (unsigned j = 0; j < PoolSize; j++)
			{
				const float &e =  input[oi + i][oj + j];
				if (e > value) value = e;
			}
		return value;
	}

	template <unsigned PoolSize, unsigned InputRows, unsigned InputCols>
	void backward(const float input,
                   const float output,
                   const float delta_output)
	{
		return input == output ? delta_output : 0.0;
	}
};

struct average
{
	template <unsigned PoolSize, unsigned InputRows, unsigned InputCols>
	float forward(const matrix<InputRows, InputCols> &input, const unsigned oi, const unsigned oj)
	{
		float value = 0.0;
		for (unsigned i = 0; i < PoolSize; i++)
			for (unsigned j = 0; j < PoolSize; j++)
			{
				const float &e = input[oi + i][oj + j];
				if (e > value) value += e;
			}
		return value / PoolSize / PoolSize;
	}

	template <unsigned PoolSize, unsigned InputRows, unsigned InputCols>
	float backward(const float input,
                   const float output,
                   const float delta_output)
	{
		return delta_output / PoolSize / PoolSize;
	}
};

} // pooling_methods

// todo: test
template <unsigned PoolSize, typename PoolMethod>
struct pooling
{
	static_assert(PoolSize > 1);

	template <typename InputShape>
	struct type;

	template <unsigned InputRows, unsigned InputCols>
	struct type<shape_t<InputRows, InputCols>>
	{
		static_assert(InputRows % PoolSize == 0 && InputCols % PoolSize == 0);

		static constexpr unsigned OutputRows = InputRows / PoolSize;
		static constexpr unsigned OutputCols = InputCols / PoolSize;

		using output_shape = shape_t<OutputRows, OutputCols>;

		struct params_t;

		static void forward(const matrix<InputRows, InputCols> &input,
                            matrix<OutputRows, OutputCols> &output)
		{
			for (unsigned i = 0; i < OutputRows; i++)
				for (unsigned j = 0; j < OutputCols; j++)
					output[i][j] = PoolMethod::forward<PoolSize>(input, i*PoolSize, j*PoolSize);
		}

		static void backward(const matrix<InputRows, InputCols> &input,
                             const matrix<InputRows, InputCols> &output,
                             matrix<OutputRows, OutputCols> &delta_input,
                             const matrix<OutputRows, OutputCols> &delta_output)
		{
			for (unsigned i = 0; i < InputRows; i++)
				for (unsigned j = 0; j < InputCols; j++)
					{
						delta_input[i][j] = PoolMethod::backward<PoolSize>(
							input[i][j],
							output[i / PoolSize][j / PoolSize],
							delta_output[i / PoolSize][j / PoolSize]
						);
					}
		}
	};
};

// todo: actually test this
template <unsigned KernelCount, unsigned KernelSize, unsigned Stride = 1>
struct convolution
{
	static_assert(KernelCount > 0, "kernel count can't be zero, you nonse");
	static_assert(KernelSize > 0, "kernel size can't be zero, you cretin");
	static_assert(Stride > 0, "no, the stride can't be zero, you fucking spoon");

	template <typename InputShape>
	struct type;

	template <unsigned InputRows, unsigned InputCols>
	struct type<shape_t<InputRows, InputCols>>
	{
		// might need to remove this restriction but im not quite there yet
		static_assert(InputRows - KernelSize % Stride == 0, "Input rows are not fully covered by the set kernel size and stride");
		static_assert(InputCols - KernelSize % Stride == 0, "Input columns are not fully covered by the set kernel size and stride");

		static constexpr unsigned OutputRows = (InputRows - KernelSize) / Stride + 1;
		static constexpr unsigned OutputCols = (InputCols - KernelSize) / Stride + 1;

		using output_shape = shape_t<KernelCount, OutputRows, OutputCols>;

		struct params_t
		{
			tensor<shape_t<KernelCount, KernelSize, KernelSize>> kernels;

			void randomise()
			{
				nn::util::randomise(kernels) /= KernelSize * KernelSize;
			}
		};

		static void forward(const matrix<InputRows, InputCols> &input,
                            tensor<output_shape> &output,
                            const params_t& params)
		{
			for (unsigned kn = 0; kn < KernelCount; kn++)
			{
				const auto &kernel = params.kernels[kn];

				for (unsigned i = 0; i < OutputRows; i++)
					for (unsigned j = 0; j < OutputCols; j++)
					{
						float &v = output[kn][i][j];

						for (unsigned ki = 0; ki < KernelSize; ki++)
							for (unsigned kj = 0; kj < KernelSize; kj++)
								v += kernel[ki][kj] * input[i*Stride + ki][j*Stride + kj];
					}
			}
		}

		static void backward(const matrix<InputRows, InputCols> &input,
                             const tensor<output_shape> &output,
                             const params_t &params,
                             matrix<InputRows, InputCols> &delta_input,
                             const tensor<output_shape> &delta_output,
                             params_t &delta_params)
		{
			for (unsigned i = 0; i < InputRows; i++)
				for (unsigned j = 0; j < InputCols; j++)
				{
					float &v = delta_input[i][j] = 0.0f;

					for (unsigned ki = 0; ki < KernelSize; ki += Stride)
						for (unsigned kj = 0; kj < KernelSize; kj += Stride)
						{
							if (i < ki || j < kj)
								continue;

							for (unsigned kn = 0; kn < KernelCount; kn++)
								v += delta_output[kn][i - ki][j - kj] * params.kernels[kn][ki][kj];
						}
				}

			for (unsigned kn = 0; kn < KernelCount; kn++)
			{
				const auto &kernel = params.kernels[kn];
				auto &delta_kernel = delta_params.kernels[kn];
				const auto &delta_output_k = delta_output[kn];

				for (unsigned ki = 0; ki < KernelSize; ki++)
					for (unsigned kj = 0; kj < KernelSize; kj++)
					{
						float &v = delta_kernel[ki][kj];

						for (unsigned i = 0; i < OutputRows; i++)
							for (unsigned j = 0; j < OutputCols; j++)
								v += delta_output_k[i][j] * input[i * Stride + ki][j * Stride + kj];
					}
			}
		}
	};
};

} // layers

} // nn