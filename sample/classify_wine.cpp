#include "nn/nn.hpp"

/* So there's a super simple data set on the UCI Machine Learning
 * Repository that you can check out here:
 *
 *  https://archive.ics.uci.edu/ml/datasets/Wine
 *
 * It's a data set of 13 data points and 3 classes that make a
 * really good practise set for simple classification problems.
 *
 * This is a bare bones solution for it, will be adding better
 * training methods soon.
*/

constexpr size_t NUM_SAMPLES = 178;
constexpr size_t NUM_DATA_POINTS = 13;
constexpr size_t NUM_CLASSES = 3;

using MyNetwork = nn::Network<NUM_DATA_POINTS,
							  nn::FullyConnectedLayer<8>::Type,
							  nn::NonLinearityLayer<nn::LogisticFunction>::Type,
							  nn::FullyConnectedLayer<NUM_CLASSES>::Type,
							  nn::SoftmaxLayer>;

// defined globally, as these could easily wreck the stack
nn::Forward<NUM_SAMPLES, MyNetwork> fwd, deltaFwd;
la::Matrix<NUM_SAMPLES, NUM_CLASSES> expectation;
nn::Params<MyNetwork> params, deltaParams;
std::array<int, NUM_SAMPLES> labels = {};
std::array<int, NUM_SAMPLES> predicted_labels = {};

int main(int argc, const char *argv[])
{
	{
		FILE *file;
		if (0 != fopen_s(&file, "sample\\wine.data", "r"))
		{
			puts("failed to open file");
			return 1;
		}

		for (size_t n = 0; n < NUM_SAMPLES; n++)
		{
			fscanf_s(file, "%d,", &labels[n]);
			labels[n]--;
			nn::util::extract_expectation(labels[n], expectation[n]);

			for (size_t m = 0; m < NUM_DATA_POINTS; m++)
				fscanf_s(file, "%lf,", &fwd.input[n][m]);
		}

		fclose(file);
	}

	nn::util::normalise(fwd.input);

	nn::util::randomise(params);

	// trivial gradient descent, better training models to come
	for (size_t epoch = 0; epoch < 100; epoch++)
	{
		const double cost = nn::cost<nn::CrossEntropyCostFunction>(expectation, fwd, params);

		printf("%3zd, cost: %.5lf\n", epoch, cost);

		nn::backward<nn::CrossEntropyCostFunction>(expectation, fwd, params, deltaFwd, deltaParams);

		deltaParams *= 0.1;
		params -= deltaParams;
	}

	const auto& output = nn::forward(fwd, params);

	const double cost = nn::cost<nn::CrossEntropyCostFunction>(expectation, output);

	printf("final cost: %.5lf\n", cost);

	nn::util::classify(output, predicted_labels);

	size_t confusion[NUM_CLASSES][NUM_CLASSES] = {};
	nn::util::confusion(confusion, labels, predicted_labels);

	// Print the confusion matrix to get a good idea of our accuracy
	for (size_t j = 0; j < NUM_CLASSES; j++)
	{
		for (size_t i = 0; i < NUM_CLASSES; i++)
			printf("% 2zd, ", confusion[j][i]);

		putchar('\n');
	}

	return 0;
}