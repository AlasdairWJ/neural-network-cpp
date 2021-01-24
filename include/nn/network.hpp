#include <type_traits>

#include "la.hpp"
#include "math.hpp"
#include "util.hpp"

namespace nn
{

template<typename, typename = void>
constexpr bool is_type_complete_v = false;

template<typename T>
constexpr bool is_type_complete_v<T, std::void_t<decltype(sizeof(T))>> = true;

// -----------------------------------------------------------------------------

// this should be doable as a variable, but how??
template <typename LayerType>
constexpr size_t ParamCount()
{
	if constexpr(is_type_complete_v<typename LayerType::Params>)
		return LayerType::Params::Count;
	else
		return 0;
}

// -----------------------------------------------------------------------------

template <
	size_t InputSize,
	template <size_t> typename ...LayerTypes
>
struct network_t;

template <
	size_t InputSize,
	template <size_t> typename TheLayerType
>
struct network_t<InputSize, TheLayerType>
{
	static constexpr bool IsFinalLayer = true;

	using LayerType = TheLayerType<InputSize>;

	static constexpr size_t OutputSize = LayerType::OutputSize;

	static constexpr size_t FinalOutputSize = OutputSize;

	static constexpr size_t ParamCount = ParamCount<LayerType>();
};

template <
	size_t InputSize,
	template <size_t> typename TheLayerType,
	template <size_t> typename NextLayerType,
	template <size_t> typename ...RestLayerTypes
>
struct network_t<InputSize, TheLayerType, NextLayerType, RestLayerTypes...>
{
	static constexpr bool IsFinalLayer = false;

	using LayerType = TheLayerType<InputSize>;

	static constexpr size_t OutputSize = LayerType::OutputSize;

	using NextNetworkType = network_t<OutputSize, NextLayerType, RestLayerTypes...>;

	static constexpr size_t FinalOutputSize = NextNetworkType::FinalOutputSize;

	static constexpr size_t ParamCount = ParamCount<LayerType>() + NextNetworkType::ParamCount;
};

// -----------------------------------------------------------------------------

template <size_t N, typename NetworkType>
struct forward_t;

template <
	size_t N,
	size_t InputSize,
	template <size_t> typename LayerType
>
struct forward_t<N, network_t<InputSize, LayerType>>
{
	la::matrix<N, InputSize> input;
	la::matrix<N, LayerType<InputSize>::OutputSize> output;

	auto& get_next() { return output; }
	const auto& get_next() const { return output; }

	auto& get_output() { return output; }
	const auto& get_output() const { return output; }
};

template <
	size_t N,
	size_t InputSize,
	template <size_t> typename LayerType,
	template <size_t> typename NextLayerType,
	template <size_t> typename ...RestLayerTypes
>
struct forward_t<N, network_t<InputSize, LayerType, NextLayerType, RestLayerTypes...>>
{
	la::matrix<N, InputSize> input;

	forward_t<N, network_t<LayerType<InputSize>::OutputSize, NextLayerType, RestLayerTypes...>> next;

	auto& get_next() { return next.input; }
	const auto& get_next() const { return next.input; }

	auto& get_output() { return next.getOutput(); }
	const auto& get_output() const { return next.getOutput(); }
};

// -----------------------------------------------------------------------------

template <typename NetworkType>
using params_t = la::vector<NetworkType::ParamCount + 1>;

template <size_t N, typename NetworkType>
using output_t = la::matrix<N, NetworkType::FinalOutputSize>;

// -----------------------------------------------------------------------------

// params should randomise themselves, as suitable param ranges will vary
template <typename NetworkType>
void randomise_params(params_t<NetworkType>& params)
{
	if constexpr(ParamCount<NetworkType::LayerType>() != 0)
		NetworkType::LayerType::Params::randomise(reinterpret_cast<typename NetworkType::LayerType::Params&>(params));
	if constexpr(!NetworkType::IsFinalLayer)
		randomise_params<NetworkType::NextNetworkType>(params.offset<ParamCount<NetworkType::LayerType>()>());
}

// -----------------------------------------------------------------------------

template <size_t N, typename NetworkType>
auto forward(forward_t<N, NetworkType>& fwd,
             const params_t<NetworkType>& params) -> const output_t<N, NetworkType>&
{
	if constexpr(ParamCount<NetworkType::LayerType>() == 0)
	{
		auto& output = fwd.get_next();
		for (size_t n = 0; n < N; n++)
			NetworkType::LayerType::forward(fwd.input[n], output[n]);
	}
	else
	{
		const auto& layerParams = reinterpret_cast<const NetworkType::LayerType::Params&>(params);

		auto& output = fwd.get_next();
		for (size_t n = 0; n < N; n++)
			NetworkType::LayerType::forward(fwd.input[n], output[n], layerParams);
	}


	if constexpr(NetworkType::IsFinalLayer)
	{
		return fwd.output;
	}
	else
	{
		return forward(fwd.next, params.offset<ParamCount<NetworkType::LayerType>()>());
	}
}

// -----------------------------------------------------------------------------

template <typename CostFunctionType, size_t N, size_t OutputSize>
double cost(const la::matrix<N, OutputSize>& expectation,
            const la::matrix<N, OutputSize>& prediction)
{
	double value = 0.0;
	for (size_t n = 0; n < N; n++)
		value += CostFunctionType::cost(expectation[n], prediction[n]);
	return value / N;
}

template <typename CostFunctionType, size_t N, typename NetworkType>
double cost(const output_t<N, NetworkType>& expectation,
            forward_t<N, NetworkType>& fwd,
            const params_t<NetworkType>& params)
{
	return cost<CostFunctionType>(expectation, forward(fwd, params));
}

// -----------------------------------------------------------------------------

template <typename CostFunctionType, size_t N, typename NetworkType>
auto backward(const output_t<N, NetworkType>& expectation,
              const forward_t<N, NetworkType>& fwd,
              const params_t<NetworkType>& params,
              forward_t<N, NetworkType>& delta_fwd,
              params_t<NetworkType>& delta_params) -> decltype(delta_params)
{
	if constexpr(NetworkType::IsFinalLayer)
	{
		for (size_t n = 0; n < N; n++)
			CostFunctionType::derivative(expectation[n], fwd.output[n], delta_fwd.output[n]);
	}
	else
	{
		backward<CostFunctionType>(
			expectation,
			fwd.next,
			params.offset<ParamCount<NetworkType::LayerType>()>(),
			delta_fwd.next,
			delta_params.offset<ParamCount<NetworkType::LayerType>()>()
		);
	}

	const auto &output = fwd.get_next();
	const auto &delta_output = delta_fwd.get_next();

	if constexpr(ParamCount<NetworkType::LayerType>() == 0)
	{
		for (size_t n = 0; n < N; n++)
		{
			NetworkType::LayerType::backward(
				fwd.input[n],
				output[n],
				delta_fwd.input[n],
				delta_output[n]
			);
		}
	}
	else
	{
		auto& this_delta_params = delta_params.truncate<ParamCount<NetworkType::LayerType>()>();

		this_delta_params.zero();

		for (size_t n = 0; n < N; n++)
		{
			NetworkType::LayerType::backward(
				fwd.input[n],
				output[n],
				reinterpret_cast<const NetworkType::LayerType::Params&>(params),
				delta_fwd.input[n],
				delta_output[n],
				reinterpret_cast<typename NetworkType::LayerType::Params&>(delta_params)
			);
		}

		this_delta_params /= N;
	}

	return delta_params;
}

// -----------------------------------------------------------------------------

template <typename CostFunctionType, size_t N, typename NetworkType>
auto numerical_gradient(const la::matrix<N, NetworkType::FinalOutputSize>& expectation,
                        forward_t<N, NetworkType>& fwd,
                        params_t<NetworkType>& params,
                        params_t<NetworkType>& delta_params) -> decltype(delta_params)
{
	const double eps = 1e-5;

	for (size_t i = 0; i < NetworkType::NumParams; i++)
	{
		const double param_value = params[i];

		params[i] = param_value + eps;
		const double cost_plus = cost<CostFunctionType>(expectation, fwd, params);

		params[i] = param_value - eps;
		const double cost_minus = cost<CostFunctionType>(expectation, fwd, params);

		params[i] = param_value;

		delta_params[i] = 0.5 * (cost_plus - cost_minus) / eps;
	}

	return delta_params;
}

}