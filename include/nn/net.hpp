#include <type_traits>

#include "la.hpp"
#include "math.hpp"
#include "util.hpp"

namespace nn
{

template <
	size_t InputSize,
	template <size_t> typename ...LayerTypes
>
struct Network;

template <
	size_t InputSize,
	template <size_t> typename TheLayerType
>
struct Network<InputSize, TheLayerType>
{
	static constexpr bool IsFinalLayer = true;

	using LayerType = TheLayerType<InputSize>;

	static constexpr size_t OutputSize = LayerType::OutputSize;

	static constexpr size_t FinalOutputSize = OutputSize;

	static constexpr size_t NumParams = LayerType::Params::Count;
};

template <
	size_t InputSize,
	template <size_t> typename TheLayerType,
	template <size_t> typename NextLayerType,
	template <size_t> typename ...RestLayerTypes
>
struct Network<InputSize, TheLayerType, NextLayerType, RestLayerTypes...>
{
	static constexpr bool IsFinalLayer = false;

	using LayerType = TheLayerType<InputSize>;

	static constexpr size_t OutputSize = LayerType::OutputSize;

	using NextNetworkType = Network<OutputSize, NextLayerType, RestLayerTypes...>;

	static constexpr size_t FinalOutputSize = NextNetworkType::FinalOutputSize;

	static constexpr size_t NumParams = LayerType::Params::Count + NextNetworkType::NumParams;
};

// -----------------------------------------------------------------------------

template <size_t N, typename NetworkType>
struct Forward;

template <
	size_t N,
	size_t InputSize,
	template <size_t> typename LayerType
>
struct Forward<N, Network<InputSize, LayerType>>
{
	la::Matrix<N, InputSize> input;
	la::Matrix<N, LayerType<InputSize>::OutputSize> output;

	auto& getNext() { return output; }
	const auto& getNext() const { return output; }

	auto& getOutput() { return output; }
	const auto& getOutput() const { return output; }
};

template <
	size_t N,
	size_t InputSize,
	template <size_t> typename LayerType,
	template <size_t> typename NextLayerType,
	template <size_t> typename ...RestLayerTypes
>
struct Forward<N, Network<InputSize, LayerType, NextLayerType, RestLayerTypes...>>
{
	la::Matrix<N, InputSize> input;

	Forward<N, Network<LayerType<InputSize>::OutputSize, NextLayerType, RestLayerTypes...>> next;

	auto& getNext() { return next.input; }
	const auto& getNext() const { return next.input; }

	auto& getOutput() { return next.getOutput(); }
	const auto& getOutput() const { return next.getOutput(); }
};

// -----------------------------------------------------------------------------

template <typename NetworkType>
using Params = la::Vector<NetworkType::NumParams + 1>;
// plus one to stop rogue zero length vectors

template <size_t N, typename NetworkType>
using Output = la::Matrix<N, NetworkType::FinalOutputSize>;

// -----------------------------------------------------------------------------

template <size_t N, typename NetworkType>
const Output<N, NetworkType>& forward(Forward<N, NetworkType>& fwd,
									  const Params<NetworkType>& params)
{
	if constexpr(NetworkType::LayerType::Params::Count == 0)
	{
		auto& output = fwd.getNext();
		for (size_t n = 0; n < N; n++)
			NetworkType::LayerType::forward(fwd.input[n], output[n]);
	}
	else
	{
		const auto& layerParams = reinterpret_cast<const NetworkType::LayerType::Params&>(params);

		auto& output = fwd.getNext();
		for (size_t n = 0; n < N; n++)
			NetworkType::LayerType::forward(fwd.input[n], output[n], layerParams);
	}


	if constexpr(NetworkType::IsFinalLayer)
	{
		return fwd.output;
	}
	else
	{
		return forward(fwd.next, params.offset<NetworkType::LayerType::Params::Count>());
	}
}

// -----------------------------------------------------------------------------

template <typename CostFunctionType, size_t N, size_t OutputSize>
double cost(const la::Matrix<N, OutputSize>& expectation,
			const la::Matrix<N, OutputSize>& prediction)
{
	double value = 0.0;
	for (size_t n = 0; n < N; n++)
		value += CostFunctionType::cost(expectation[n], prediction[n]);
	return value / N;
}

template <typename CostFunctionType, size_t N, typename NetworkType>
double cost(const la::Matrix<N, NetworkType::FinalOutputSize>& expectation,
			Forward<N, NetworkType>& fwd,
			const Params<NetworkType>& params)
{
	return cost<CostFunctionType>(expectation, forward(fwd, params));
}

// -----------------------------------------------------------------------------

template <typename CostFunctionType, size_t N, typename NetworkType>
auto backward(const la::Matrix<N, NetworkType::FinalOutputSize>& expectation,
			  const Forward<N, NetworkType>& fwd,
			  const Params<NetworkType>& params,
			  Forward<N, NetworkType>& deltaFwd,
			  Params<NetworkType>& deltaParams) -> decltype(deltaParams)
{
	if constexpr(NetworkType::IsFinalLayer)
	{
		for (size_t n = 0; n < N; n++)
			CostFunctionType::derivative(expectation[n], fwd.output[n], deltaFwd.output[n]);
	}
	else
	{
		backward<CostFunctionType>(
			expectation,
			fwd.next,
			params.offset<NetworkType::LayerType::Params::Count>(),
			deltaFwd.next,
			deltaParams.offset<NetworkType::LayerType::Params::Count>()
		);
	}

	if constexpr(NetworkType::LayerType::Params::Count == 0)
	{
		const auto& output = fwd.getNext();
		const auto& deltaOutput = deltaFwd.getNext();

		for (size_t n = 0; n < N; n++)
		{
			NetworkType::LayerType::backward(
				fwd.input[n],
				output[n],
				deltaFwd.input[n],
				deltaOutput[n]
			);
		}
	}
	else
	{
		const auto& output = fwd.getNext();
		const auto& deltaOutput = deltaFwd.getNext();

		la::Vector<NetworkType::LayerType::Params::Count> deltaParamsBuffer;

		auto& thisDeltaParams = deltaParams.truncate<NetworkType::LayerType::Params::Count>();

		thisDeltaParams.zero();

		for (size_t n = 0; n < N; n++)
		{
			NetworkType::LayerType::backward(
				fwd.input[n],
				output[n],
				reinterpret_cast<const NetworkType::LayerType::Params&>(params),
				deltaFwd.input[n],
				deltaOutput[n],
				reinterpret_cast<typename NetworkType::LayerType::Params&>(deltaParamsBuffer)
			);

			thisDeltaParams += deltaParamsBuffer;
		}
		
		thisDeltaParams /= N;
	}

	return deltaParams;
}

// -----------------------------------------------------------------------------

template <typename CostFunctionType, size_t N, typename NetworkType>
auto numerical_gradient(const la::Matrix<N, NetworkType::FinalOutputSize>& expectation,
						Forward<N, NetworkType>& fwd,
						Params<NetworkType>& params,
						Params<NetworkType>& deltaParams) -> decltype(deltaParams)
{
	const double eps = 1e-5;

	for (size_t i = 0; i < NetworkType::NumParams; i++)
	{
		const double originalParamValue = params[i];

		params[i] = originalParamValue + eps;
		const double cost_plus = cost<CostFunctionType>(expectation, fwd, params);

		params[i] = originalParamValue - eps;
		const double cost_minus = cost<CostFunctionType>(expectation, fwd, params);

		params[i] = originalParamValue;

		deltaParams[i] = 0.5 * (cost_plus - cost_minus) / eps;
	}

	return deltaParams;
}

}