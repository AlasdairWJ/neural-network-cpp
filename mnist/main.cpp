#include "cnn/cnn.hpp"
#include "mnist.hpp"

#include <memory>

constexpr unsigned NUM_TRAINING_SAMPLES = 60'000;
constexpr unsigned NUM_TEST_SAMPLES = 10'000;
constexpr unsigned BATCH_SIZE = 100;
constexpr unsigned NUM_CLASSES = 10;
constexpr unsigned IMAGE_SIZE = 28;

static_assert(NUM_TRAINING_SAMPLES % BATCH_SIZE == 0, "batch size must perfectly divisible by the number of training samples");

using InputShape = shape_t<IMAGE_SIZE, IMAGE_SIZE>;
using OutputShape = shape_t<NUM_CLASSES>;

/*
// convolutional/pooling layers aren't quite ready yet
using MyNetwork = nn::network_t<
	InputShape,
	nn::layers::convolution<32, 3>::type,
	nn::layers::relu,
//	nn::layers::max_pooling<2>::type,
	nn::layers::fully_connected<NUM_CLASSES>::type,
	nn::layers::softmax>;
*/

using MyNetwork = nn::network_t<
	InputShape,
	nn::layers::fully_connected<30>::type,
	nn::layers::logistic,
	nn::layers::fully_connected<NUM_CLASSES>::type,
	nn::layers::softmax>;

// these objects are not containers with pointers to the heap, they ARE the data, and they're big.
// wrapping it in a structure to keep it off the stack and to keep it out of global.
struct program
{
	uint8_t raw_training_labels[NUM_TRAINING_SAMPLES];
	uint8_t raw_training_images[NUM_TRAINING_SAMPLES][IMAGE_SIZE][IMAGE_SIZE];

	uint8_t raw_test_labels[NUM_TEST_SAMPLES];
	uint8_t raw_test_images[NUM_TEST_SAMPLES][IMAGE_SIZE][IMAGE_SIZE];

	unsigned shuffled_indices[NUM_TRAINING_SAMPLES];

	vector_of<NUM_TRAINING_SAMPLES, InputShape> training_images;
	vector_of<NUM_TRAINING_SAMPLES, OutputShape> training_expectation;

	vector_of<NUM_TEST_SAMPLES, OutputShape> test_expectation;

	vector_of<BATCH_SIZE, OutputShape> batch_expectation;

	nn::params_t<MyNetwork> params, gradient, velocity;
	nn::forward_t<BATCH_SIZE, MyNetwork> fwd, delta_fwd;

	nn::forward_t<NUM_TEST_SAMPLES, MyNetwork> test_fwd;

	int run(int argc, const char *argv[]);
};

int program::run(const int argc, const char *argv[])
{
	// LOAD TRAINING DATA

	if (!load_idx("data\\train-labels.idx1-ubyte", raw_training_labels))
	{
		puts("failed to load label file");
		return 1;
	}

	nn::util::expectation_from_labels(raw_training_labels, training_expectation);

	if (!load_idx("data\\train-images.idx3-ubyte", raw_training_images))
	{
		puts("failed to load image file");
		return 1;
	}

	for (unsigned n = 0; n < NUM_TRAINING_SAMPLES; n++)
	{
		for (unsigned i = 0; i < IMAGE_SIZE; i++)
			for (unsigned j = 0; j < IMAGE_SIZE; j++)
				training_images[n][i][j] = static_cast<float>(raw_training_images[n][i][j]) / 255.0f;
	}

	// LOAD TEST DATA

	if (!load_idx("data\\t10k-labels.idx1-ubyte", raw_test_labels))
	{
		puts("failed to load label file");
		return 1;
	}

	nn::util::expectation_from_labels(raw_test_labels, test_expectation);

	if (!load_idx("data\\t10k-images.idx3-ubyte", raw_test_images))
	{
		puts("failed to load image file");
		return 1;
	}

	auto &test_images = test_fwd.input;

	for (unsigned n = 0; n < NUM_TEST_SAMPLES; n++)
	{
		for (unsigned i = 0; i < IMAGE_SIZE; i++)
			for (unsigned j = 0; j < IMAGE_SIZE; j++)
				test_images[n][i][j] = static_cast<float>(raw_test_images[n][i][j]) / 255.0f;
	}

	// DO STUFF

	for (unsigned ix = 0; ix < NUM_TRAINING_SAMPLES; ix++)
		shuffled_indices[ix] = ix;

	auto &batch_images = fwd.input;

	nn::randomise_params<MyNetwork>(params);
	velocity = 0.0f;

	const float decay = 0.9f;
	const float learning_rate = 0.1f;

	for (unsigned epoch = 0; epoch < 10; epoch++)
	{
		printf("starting training epoch #%u...", epoch);

		nn::util::shuffle(shuffled_indices);

		unsigned ix = 0;
		for (unsigned iteration = 0; iteration < NUM_TRAINING_SAMPLES / BATCH_SIZE; iteration++)
		{
			for (unsigned n = 0; n < BATCH_SIZE; n++)
			{
				batch_images[n] = training_images[shuffled_indices[ix]];
				batch_expectation[n] = training_expectation[shuffled_indices[ix]];
				ix++;
			}

			const auto &batch_prediction = nn::forward(fwd, params);

			const float j = nn::cost<nn::cost_functions::cross_entropy>(batch_expectation, batch_prediction);

			nn::backward<nn::cost_functions::cross_entropy>(batch_expectation, fwd, params, delta_fwd, gradient);

			velocity *= decay;
			gradient *= learning_rate;
			velocity -= gradient;

			params += velocity;
		}

		// test set
		{
			const auto &prediction = nn::forward(test_fwd, params);

			unsigned correct = 0;
			for (unsigned n = 0; n < NUM_TEST_SAMPLES; n++)
			{
				if (nn::util::classify(prediction[n]) == nn::util::classify(test_expectation[n]))
					correct++;
			}

			printf("test accuracy: %.3f\n", 100.0f * static_cast<float>(correct) / NUM_TEST_SAMPLES);
		}
	}

	nn::util::save("params.dat", params);

	return 0;
}

int main(const int argc, const char* argv[])
{
	auto p = std::make_unique<program>();
	return p->run(argc, argv);
}