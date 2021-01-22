#include "nn.hpp"
#include "layers.hpp"

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
size_t labels[NUM_SAMPLES] = {};
size_t predicted_labels[NUM_SAMPLES] = {};

int main(int argc, const char *argv[])
{
	{
		FILE *file;
		if (0 != fopen_s(&file, "wine.data", "r"))
		{
			puts("failed to open file");
			return 1;
		}

		for (size_t n = 0; n < NUM_SAMPLES; n++)
		{
			fscanf_s(file, "%d,", &labels[n]);
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

		printf("%3d, cost: %.5lf\n", epoch, cost);

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

	for (size_t j = 0; j < NUM_CLASSES; j++)
	{
		for (size_t i = 0; i < NUM_CLASSES; i++)
			printf("% 2d, ", confusion[j][i]);
		
		putchar('\n');
	}
	
	return 0;
}