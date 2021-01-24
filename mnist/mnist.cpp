#include <cstdint>
#include <cstdlib> // _byteswap_ulong
#include <memory>

#include "nn/nn.hpp"

// -----------------------------------------------------------------------------

template <size_t N>
bool load_label_file(const char* filename, std::array<uint8_t, N>& labels);

template <size_t N, size_t D>
bool load_image_file(const char* filename, la::matrix<N, D>& values);

// -----------------------------------------------------------------------------

constexpr size_t NUM_SAMPLES = 60'000;
constexpr size_t NUM_TEST_SAMPLES = 10'000;
constexpr size_t BATCH_SIZE = 200;
constexpr size_t NUM_DATA_POINTS = 28*28;
constexpr size_t NUM_CLASSES = 10;

using MyNetwork = nn::network_t<NUM_DATA_POINTS,
                                nn::layers::fully_connected<300>::type,
                                nn::layers::logistic,
                                nn::layers::fully_connected<NUM_CLASSES>::type,
                                nn::layers::softmax>;

// -----------------------------------------------------------------------------

int main(int argc, const char *argv[])
{
	auto params = std::make_unique<nn::params_t<MyNetwork>>();
	nn::randomise_params<MyNetwork>(*params);

	// train
	{
		puts("loading training labels...");

		auto labels = std::make_unique<std::array<uint8_t, NUM_SAMPLES>>();
		auto expectation = std::make_unique<la::matrix<NUM_SAMPLES, NUM_CLASSES>>();

		if (!load_label_file("train-labels.idx1-ubyte", *labels))
		{
			puts("failed to read label data");
			return 1;
		}

		nn::util::extract_expectation(*labels, *expectation);

		// ----

		puts("loading training images...");

		auto input = std::make_unique<la::matrix<NUM_SAMPLES, NUM_DATA_POINTS>>();

		if (!load_image_file("train-images.idx3-ubyte", *input))
		{
			puts("failed to read label data");
			return 1;
		}

		puts("training...");

		auto fwd = std::make_unique<nn::forward_t<BATCH_SIZE, MyNetwork>>();
		auto batchExpectation = std::make_unique<la::matrix<BATCH_SIZE, NUM_CLASSES>>();

		auto deltaFwd = std::make_unique<nn::forward_t<BATCH_SIZE, MyNetwork>>();
		auto deltaParams = std::make_unique<nn::params_t<MyNetwork>>();
		auto velocity = std::make_unique<nn::params_t<MyNetwork>>();

		for (int epoch = 1; epoch <= 200; epoch++)
		{
			nn::util::generate_minibatch(*input, *expectation, fwd->input, *batchExpectation);

			const double cost = nn::cost<nn::cost_functions::cross_entropy>(*batchExpectation, *fwd, *params);

			printf("epoch: %3d, cost: %.5lf\n", epoch, cost);

			nn::backward<nn::cost_functions::cross_entropy>(*batchExpectation, *fwd, *params, *deltaFwd, *deltaParams);

			(*velocity) *= 0.9; // decay
			(*deltaParams) *= 0.1; // learning rate;
			(*velocity) -= *deltaParams;
			(*params) += *velocity;
		}

		puts("finished training.");
	}

	// test
	{
		puts("loading test labels...");

		auto labels = std::make_unique<std::array<uint8_t, NUM_TEST_SAMPLES>>();

		if (!load_label_file("t10k-labels.idx1-ubyte", *labels))
		{
			puts("failed to read label data");
			return 1;
		}

		puts("loading test images...");

		auto fwd = std::make_unique<nn::forward_t<NUM_TEST_SAMPLES, MyNetwork>>();

		if (!load_image_file("t10k-images.idx3-ubyte", fwd->input))
		{
			puts("failed to read label data");
			return 1;
		}

		puts("classifying...");
		const auto &prediction = nn::forward(*fwd, *params);

		auto predictedLabels = std::make_unique<std::array<uint8_t, NUM_TEST_SAMPLES>>();
		nn::util::classify(prediction, *predictedLabels);

		int confusion[NUM_CLASSES][NUM_CLASSES] = {};
		nn::util::confusion(confusion, *labels, *predictedLabels);

		puts("confusion matrix:");
		for (size_t j = 0; j < NUM_CLASSES; j++)
		{
			for (size_t i = 0; i < NUM_CLASSES; i++)
				printf("%5d, ", confusion[j][i]);

			putchar('\n');
		}

		const double accuracy = nn::util::accuracy(*labels, *predictedLabels);

		printf("accuracy: %.5lf%%\n", 100.0 * accuracy);
	}

	return 0;
}

// -----------------------------------------------------------------------------

template <size_t N>
bool load_label_file(const char* filename, std::array<uint8_t, N>& labels)
{
	FILE *file;
	if (0 != fopen_s(&file, filename, "rb"))
		return false;

	uint32_t magic_number;
	fread(&magic_number, sizeof(magic_number), 1, file);

	uint32_t label_count;
	fread(&label_count, sizeof(label_count), 1, file);

	if (_byteswap_ulong(label_count) != N)
	{
		puts("number of labels in file differs from expected value");
		printf("file has %d, expected %zd\n", label_count, N);
		return false;
	}

	fread(labels.data(), sizeof(uint8_t), N, file);

	fclose(file);

	return true;
}

// -----------------------------------------------------------------------------

template <size_t N, size_t D>
bool load_image_file(const char* filename, la::matrix<N, D>& values)
{
	FILE *file;
	if (0 != fopen_s(&file, filename, "rb"))
		return false;

	uint32_t magic_number;
	fread(&magic_number, sizeof(magic_number), 1, file);

	uint32_t image_count;
	fread(&image_count, sizeof(image_count), 1, file);

	if (_byteswap_ulong(image_count) != N)
	{
		puts("number of labels in file differs from expected value");
		printf("file has %d, expected %zd\n", image_count, N);
		return fclose(file), false;
	}

	uint32_t row_count;
	fread(&row_count, sizeof(row_count), 1, file);

	uint32_t col_count;
	fread(&col_count, sizeof(col_count), 1, file);

	if (static_cast<size_t>(_byteswap_ulong(row_count)) * _byteswap_ulong(col_count) != D)
	{
		puts("bad dimensions");
		printf("got %dx%d, expected product to be %zd\n", _byteswap_ulong(row_count), _byteswap_ulong(col_count), D);
		return fclose(file), false;
	}

	for (size_t n = 0; n < N; n++)
	{
		uint8_t image[D];
		fread(image, sizeof(uint8_t), D, file);

		for (size_t d = 0; d < D; d++)
			values[n][d] = static_cast<double>(image[d]) / 255.0;
	}

	return fclose(file), true;
}