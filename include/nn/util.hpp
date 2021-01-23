#include <random>
#include <chrono>

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

template <size_t N, size_t M>
auto classify(const la::Matrix<N, M> &prediction, size_t (&labels)[N]) -> decltype(labels)
{
	for (size_t n = 0; n < N; n++)
		labels[n] = math::argmax(prediction[n]) + 1;
	return labels;
}

template <size_t M>
auto extract_expectation(const size_t label, la::Vector<M> &expectation) -> decltype(expectation)
{
	for (size_t m = 0; m < M; m++)
		expectation[m] = math::kdelta(m, label - 1);
	return expectation;
}

template <size_t N, size_t M>
auto extract_expectation(const size_t (labels)[N], la::Matrix<N, M> &expectation) -> decltype(expectation)
{
	for (size_t n = 0; n < N; n++)
		extract_expectation(labels[n], extract_expectation[n]);
	return expectation;
}

template <size_t N, size_t M>
void confusion(size_t (&conf)[M][M], size_t (&expected_labels)[N], size_t (&predicted_labels)[N])
{
	for (size_t n = 0; n < N; n++)
		conf[expected_labels[n] - 1][predicted_labels[n] - 1]++;
}

}

} // nn