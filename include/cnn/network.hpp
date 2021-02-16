#include <type_traits>

#include "tensor.hpp"
#include "util.hpp"

namespace nn
{

// -----------------------------------------------------------------------------

// cool thing i stole

template<typename, typename = void>
constexpr bool is_type_complete_v = false;

template<typename T>
constexpr bool is_type_complete_v<T, std::void_t<decltype(sizeof(T))>> = true;

// -----------------------------------------------------------------------------

template<typename LayerType>
constexpr bool has_params_v = is_type_complete_v<LayerType::params_t>;

template<typename LayerType, typename = void>
constexpr unsigned layer_param_count_v = 0;

template<typename LayerType>
constexpr unsigned layer_param_count_v<LayerType, std::enable_if_t<has_params_v<LayerType>>> = sizeof(LayerType::params_t) / sizeof(float);

// -----------------------------------------------------------------------------

template<typename NetworkType, typename = void>
constexpr unsigned param_count_v = layer_param_count_v<NetworkType::layer>;

template<typename NetworkType>
constexpr unsigned param_count_v<NetworkType, std::enable_if_t<!NetworkType::is_final_layer>> = layer_param_count_v<NetworkType::layer> + param_count_v<NetworkType::next_network_t>;

// -----------------------------------------------------------------------------

template <
	typename InputShape,
	template <typename> typename ...LayerTypes
>
struct network_t;

template <
	typename InputShape,
	template <typename> typename LayerType
>
struct network_t<InputShape, LayerType>
{
	static constexpr bool is_final_layer = true;

	using layer = LayerType<InputShape>;

	using output_shape = typename layer::output_shape;
};

template <
	typename InputShape,
	template <typename> typename LayerType,
	template <typename> typename NextLayerType,
	template <typename> typename ...RestLayerTypes
>
struct network_t<InputShape, LayerType, NextLayerType, RestLayerTypes...>
{
	static constexpr bool is_final_layer = false;

	using layer = LayerType<InputShape>;

	using next_network_t = network_t<typename layer::output_shape, NextLayerType, RestLayerTypes...>;

	using output_shape = typename next_network_t::output_shape;
};

// -----------------------------------------------------------------------------

template <unsigned N, typename NetworkType>
struct forward_t;

template <
	unsigned N,
	typename InputShape,
	template <typename> typename LayerType
>
struct forward_t<N, network_t<InputShape, LayerType>>
{
	vector_of<N, InputShape> input;
	vector_of<N, typename LayerType<InputShape>::output_shape> output;

	auto &get_next() { return output; }
	const auto &get_next() const { return output; }

	auto &get_output() { return output; }
	const auto &get_output() const { return output; }
};

template <
	unsigned N,
	typename InputShape,
	template <typename> typename LayerType,
	template <typename> typename NextLayerType,
	template <typename> typename ...RestLayerTypes
>
struct forward_t<N, network_t<InputShape, LayerType, NextLayerType, RestLayerTypes...>>
{
	vector_of<N, InputShape> input;

	forward_t<N, network_t<typename LayerType<InputShape>::output_shape, NextLayerType, RestLayerTypes...>> next;

	auto &get_next() { return next.input; }
	const auto &get_next() const { return next.input; }

	auto &get_output() { return next.get_output(); }
	const auto &get_output() const { return next.get_output(); }
};

// -----------------------------------------------------------------------------

template <unsigned N, typename NetworkType>
using output_t = vector_of<N, typename NetworkType::output_shape>;

template <typename NetworkType>
using params_t = vector<param_count_v<NetworkType> + 1>;

// -----------------------------------------------------------------------------

// params should randomise themselves, as suitable param ranges will vary
template <typename NetworkType>
void randomise_params(params_t<NetworkType> &params)
{
	if constexpr(has_params_v<NetworkType::layer>)
		reinterpret_cast<typename NetworkType::layer::params_t &>(params).randomise();
	if constexpr(!NetworkType::is_final_layer)
		randomise_params<NetworkType::next_network_t>(params.offset<layer_param_count_v<NetworkType::layer>>());
}

// -----------------------------------------------------------------------------

template <unsigned N, typename NetworkType>
auto forward(forward_t<N, NetworkType> &fwd,
             const params_t<NetworkType> &params) -> const output_t<N, NetworkType> &
{
	if constexpr(has_params_v<NetworkType::layer>)
	{
		const auto &layer_params = reinterpret_cast<const NetworkType::layer::params_t &>(params);

		auto &output = fwd.get_next();
		for (unsigned n = 0; n < N; n++)
			NetworkType::layer::forward(fwd.input[n], output[n], layer_params);
	}
	else
	{
		auto &output = fwd.get_next();
		for (unsigned n = 0; n < N; n++)
			NetworkType::layer::forward(fwd.input[n], output[n]);
	}


	if constexpr(NetworkType::is_final_layer)
	{
		return fwd.output;
	}
	else
	{
		return forward(fwd.next, params.offset<layer_param_count_v<NetworkType::layer>>());
	}
}

// -----------------------------------------------------------------------------

template <typename CostFunctionType, unsigned N, unsigned... Sizes>
float cost(const tensor<shape_t<N, Sizes...>> &expectation,
           const tensor<shape_t<N, Sizes...>> &prediction)
{
	float value = 0.0;
	for (unsigned n = 0; n < N; n++)
		value += CostFunctionType::cost(expectation[n], prediction[n]);
	return value / N;
}

template <typename CostFunctionType, unsigned N, typename NetworkType>
float cost(const output_t<N, NetworkType> &expectation,
           forward_t<N, NetworkType> &fwd,
           const params_t<NetworkType> &params)
{
	return cost<CostFunctionType>(expectation, forward(fwd, params));
}

// -----------------------------------------------------------------------------

template <typename CostFunctionType, size_t N, typename NetworkType>
auto backward(const output_t<N, NetworkType> &expectation,
              const forward_t<N, NetworkType> &fwd,
              const params_t<NetworkType> &params,
              forward_t<N, NetworkType> &delta_fwd,
              params_t<NetworkType> &delta_params) -> decltype(delta_params)
{
	if constexpr(NetworkType::is_final_layer)
	{
		for (unsigned n = 0; n < N; n++)
			CostFunctionType::derivative(expectation[n], fwd.output[n], delta_fwd.output[n]);
	}
	else
	{
		backward<CostFunctionType>(
			expectation,
			fwd.next,
			params.offset<layer_param_count_v<NetworkType::layer>>(),
			delta_fwd.next,
			delta_params.offset<layer_param_count_v<NetworkType::layer>>()
		);
	}

	const auto &output = fwd.get_next();
	const auto &delta_output = delta_fwd.get_next();

	if constexpr(has_params_v<NetworkType::layer>)
	{
		auto &this_delta_params = delta_params.truncate<layer_param_count_v<NetworkType::layer>>();

		this_delta_params.zero();

		for (unsigned n = 0; n < N; n++)
		{
			NetworkType::layer::backward(
				fwd.input[n],
				output[n],
				reinterpret_cast<const NetworkType::layer::params_t &>(params),
				delta_fwd.input[n],
				delta_output[n],
				reinterpret_cast<typename NetworkType::layer::params_t &>(delta_params)
			);
		}

		this_delta_params /= N;
	}
	else
	{
		for (unsigned n = 0; n < N; n++)
		{
			NetworkType::layer::backward(
				fwd.input[n],
				output[n],
				delta_fwd.input[n],
				delta_output[n]
			);
		}
	}

	return delta_params;
}

// -----------------------------------------------------------------------------

// don't call this on big networks. just dont.
template <typename CostFunctionType, size_t N, typename NetworkType>
auto numerical_gradient(const output_t<N, NetworkType> &expectation,
                        forward_t<N, NetworkType> &fwd,
                        params_t<NetworkType> &params,
                        params_t<NetworkType> &delta_params) -> decltype(delta_params)
{
	const float eps = 1e-5f;

	for (size_t i = 0; i < nn::param_count_v<NetworkType>; i++)
	{
		const float param_value = params[i];

		params[i] = param_value + eps;
		const float cost_plus = cost<CostFunctionType>(expectation, fwd, params);

		params[i] = param_value - eps;
		const float cost_minus = cost<CostFunctionType>(expectation, fwd, params);

		params[i] = param_value;

		delta_params[i] = 0.5f * (cost_plus - cost_minus) / eps;
	}

	return delta_params;
}

}