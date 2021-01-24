#include <random>
#include <chrono>
#include <array>

#include "la.hpp"

namespace nn
{

namespace util
{

double randn(const double mu = 0.0, const double sigma = 1.0)
{
	static std::default_random_engine generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
	static std::normal_distribution<double> distribution;

	return distribution(generator) * sigma + mu;
}

template <typename T>
T rand(const T lower, const T upper)
{
	static std::default_random_engine generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
	static std::uniform_int_distribution<T> distribution;

	distribution.param(std::uniform_int_distribution<T>::param_type(lower, upper - 1));

	return distribution(generator);
}

template <size_t N>
auto randomise(la::Vector<N> &values, const double mu = 0.0, const double sigma = 1.0) -> decltype(values)
{
	for (double &value : values)
		value = randn(mu, sigma);
	return values;
}

template <size_t N, size_t M>
auto normalise(la::Matrix<N, M> &values) -> decltype(values)
{
	for (size_t m = 0; m < M; m++)
	{
		double mu = 0.0;
		for (size_t n = 0; n < N; n++)
			mu += values[n][m];
		mu /= N;

		double sigma = 0.0;
		for (size_t n = 0; n < N; n++)
		{
			double &v = (values[n][m] -= mu);
			sigma += v * v;
		}

		sigma = sqrt(sigma / (N - 1));
		for (size_t n = 0; n < N; n++)
			values[n][m] /= sigma;
	}
	return values;
}

template <typename LabelType, size_t M>
LabelType classify(const la::Vector<M> &prediction)
{
	return static_cast<LabelType>(math::argmax(prediction));
}

template <typename LabelType, size_t N, size_t M>
auto classify(const la::Matrix<N, M> &prediction, std::array<LabelType, N> &labels) -> decltype(labels)
{
	for (size_t n = 0; n < N; n++)
		labels[n] = classify<LabelType>(prediction[n]);
	return labels;
}

template <typename LabelType, size_t M>
auto extract_expectation(const LabelType label, la::Vector<M> &expectation) -> decltype(expectation)
{
	for (size_t m = 0; m < M; m++)
		expectation[m] = math::kdelta(static_cast<size_t>(label), m);
	return expectation;
}

template <typename LabelType, size_t N, size_t M>
auto extract_expectation(const std::array<LabelType, N> &labels, la::Matrix<N, M> &expectation) -> decltype(expectation)
{
	for (size_t n = 0; n < N; n++)
		extract_expectation(labels[n], expectation[n]);
	return expectation;
}

template <typename LabelType, typename T, size_t N, size_t M>
void confusion(T (&conf)[M][M], const std::array<LabelType, N> &expected_labels, const std::array<LabelType, N> &predicted_labels)
{
	for (size_t n = 0; n < N; n++)
		conf[expected_labels[n]][predicted_labels[n]]++;
}

template <typename LabelType, size_t N>
double accuracy(std::array<LabelType, N> &expected_labels, std::array<LabelType, N> &predicted_labels)
{
	int correct = 0;
	for (size_t n = 0; n < N; n++)
		if (expected_labels[n] == predicted_labels[n])
			correct++;
	return static_cast<double>(correct) / N;
}

template <size_t DataSetSize, size_t InputSize, size_t OutputSize, size_t BatchSize>
void generate_minibatch(const la::Matrix<DataSetSize, InputSize>& input,
					   const la::Matrix<DataSetSize, OutputSize>& output,
					   la::Matrix<BatchSize, InputSize>& batch_input,
					   la::Matrix<BatchSize, OutputSize>& batch_output)
{
	for (size_t n = 0; n < BatchSize; n++)
	{
		size_t index = rand<size_t>(0, DataSetSize);
		batch_input[n] = input[index];
		batch_output[n] = output[index];
	}
}

}

} // nn