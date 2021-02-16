#include <random>
#include <chrono>
#include <array>

#include "tensor.hpp"
#include "math.hpp"

namespace nn
{

namespace util
{

// i do NOT wanna have to write this anywhere else

float randn()
{
	static std::default_random_engine generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
	static std::normal_distribution<float> distribution;

	return distribution(generator);
}

template <typename T>
T rand(const T lower, const T upper)
{
	static std::default_random_engine generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
	static std::uniform_int_distribution<T> distribution;

	return distribution(generator, std::uniform_int_distribution<T>::param_type(lower, upper - 1));
}

template <unsigned N>
auto randomise(vector<N> &values) -> decltype(values)
{
	for (float &v : values)
		v = randn();
	return values;
}

template <typename Shape, typename = std::enable_if_t<Shape::dim != 1>>
auto randomise(tensor<Shape> &values) -> decltype(values)
{
	return randomise(values.unravel()), values;
}

template <unsigned N, unsigned M>
auto normalise(matrix<N, M> &values) -> decltype(values)
{
	for (unsigned m = 0; m < M; m++)
	{
		float mu = 0.0;
		for (unsigned n = 0; n < N; n++)
			mu += values[n][m];
		mu /= N;

		float sigma = 0.0;
		for (unsigned n = 0; n < N; n++)
		{
			float &v = (values[n][m] -= mu);
			sigma += v * v;
		}

		sigma = sqrt(sigma / (N - 1));
		for (unsigned n = 0; n < N; n++)
			values[n][m] /= sigma;
	}
	return values;
}

template <unsigned M>
unsigned classify(const vector<M> &prediction)
{
	return math::argmax(prediction);
}

template <typename LabelType, unsigned N, unsigned M>
auto classify(const matrix<N, M> &prediction, LabelType (&labels)[N]) -> decltype(labels)
{
	for (unsigned n = 0; n < N; n++)
		labels[n] = static_cast<LabelType>(classify(prediction[n]));
	return labels;
}

template <unsigned M>
auto expectation_from_label(const unsigned label, vector<M> &expectation) -> decltype(expectation)
{
	for (unsigned m = 0; m < M; m++)
		expectation[m] = math::kdelta(label, m);
	return expectation;
}

template <typename LabelType, unsigned N, unsigned M>
auto expectation_from_labels(const LabelType (&labels)[N], matrix<N, M> &expectation) -> decltype(expectation)
{
	for (unsigned n = 0; n < N; n++)
		expectation_from_label(static_cast<unsigned>(labels[n]), expectation[n]);
	return expectation;
}

template <unsigned N>
auto shuffle(unsigned (&indices)[N]) -> decltype(indices)
{
	for (unsigned n = 0; n+1 < N; n++)
	{
		unsigned ix = rand(n, N);
		if (ix != n)
			std::swap(indices[n], indices[ix]);
	}
	return indices;
}

/*



template <typename LabelType, typename T, unsigned N, unsigned M>
void confusion(T (&conf)[M][M], const std::array<LabelType, N> &expected_labels, const std::array<LabelType, N> &predicted_labels)
{
	for (unsigned n = 0; n < N; n++)
		conf[expected_labels[n]][predicted_labels[n]]++;
}

template <typename LabelType, unsigned N>
float accuracy(std::array<LabelType, N> &expected_labels, std::array<LabelType, N> &predicted_labels)
{
	int correct = 0;
	for (unsigned n = 0; n < N; n++)
		if (expected_labels[n] == predicted_labels[n])
			correct++;
	return static_cast<double>(correct) / N;
}
*/

template <unsigned DataSetSize, unsigned BatchSize, typename InputShape, typename OutputShape>
void generate_minibatch(
	const vector_of<DataSetSize, InputShape> &input,
	const vector_of<DataSetSize, OutputShape> &output,
	vector_of<BatchSize, InputShape> &batch_input,
	vector_of<BatchSize, OutputShape> &batch_output)
{
	for (unsigned n = 0; n < BatchSize; n++)
	{
		unsigned index = rand<unsigned>(0, DataSetSize);
		batch_input[n] = input[index];
		batch_output[n] = output[index];
	}
}

template <unsigned N>
bool load(const char *filename, vector<N> &values)
{
	FILE *file;

	if (0 != fopen_s(&file, filename, "rb"))
		return false;

	fread(values.data(), sizeof(float), N, file);

	fclose(file);

	return true;
}

template <unsigned N>
bool save(const char *filename, const vector<N> &values)
{
	FILE *file;

	if (0 != fopen_s(&file, filename, "wb"))
		return false;

	fwrite(values.data(), sizeof(float), N, file);

	fclose(file);

	return true;
}

} // util

} // nn