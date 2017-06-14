#include <iostream>
#include <cstdio>
#include <cassert>
#include <TH.h>
#include <math.h>
#include "tools/ppm.hpp"
#include "overfeat.hpp"

using namespace std;

int main(int argc, char *argv[])
{
	// read arguments
	fprintf(stdout, "%20s: %s\n", "argv0 (cmd)", argv[0]);
	if (argc < 2)
	{
		fprintf(stderr, "Missing argument : path to weight file\n");
		exit(0);
	}
	fprintf(stdout, "%20s: %s\n", "argv1 (weight file)", argv[1]);
	if (argc < 3)
	{
		fprintf(stderr, "Missing argument : (number of top classes | -1)\n");
		exit(0);
	}
	fprintf(stdout, "%20s: %s\n", "argv2 (ntop)", argv[2]);
	if (argc < 4)
	{
		fprintf(stderr, "Missing argument : network idx\n");
		exit(0);
	}
	fprintf(stdout, "%20s: %s\n", "argv3 (net idx)", argv[3]);
	int nTopClasses = atoi(argv[2]);
	int net_idx = atoi(argv[3]);
	int feature_layer;
	if (nTopClasses <= 0)
	{
		if (argc < 5)
		{
			fprintf(stderr, "Missing argument : output feature layer\n");
			exit(0);
		}
		fprintf(stdout, "%20s: %s\n", "argv4 (feat. layer)", argv[4]);
		feature_layer = atoi(argv[4]);
	}

	// initializes overfeat
	overfeat::init(argv[1], net_idx);

	THTensor *input_raw = THTensor_(new)();
	THTensor *input = THTensor_(new)();
	THTensor *probas = THTensor_(new)();

	fprintf(stdout, "Reading image ... ");

	while (readPPM(stdin, input_raw))
	{
		assert(input_raw->size[0] == 3); //input must be rgb
		int rw = input_raw->size[2], rh = input_raw->size[1];
		fprintf(stdout, "success: width=%d | height=%d\n", rw, rh);

		if (nTopClasses > 0)
		{ // print top classes
			// crop image to make it square
			int dstdim = min(rh, rw);
			THTensor_(resize3d)(input, 3, dstdim, dstdim);
			long
					sr0 = input_raw->stride[0],
					sr1 = input_raw->stride[1],
					sr2 = input_raw->stride[2],
					s0 = input->stride[0],
					s1 = input->stride[1],
					s2 = input->stride[2];

			int xoffset = 0, yoffset = 0;
			if (rh < rw)
			{
				xoffset = (rw - dstdim) / 2;
			}
			else
			{
				yoffset = (rh - dstdim) / 2;
			}

			real *data_raw = THTensor_(data)(input_raw);
			real *data = THTensor_(data)(input);

			float v_min, v_max;
			float channel_avg[3];

			for (int c = 0; c < 3; ++c)
			{
				for (int i = 0; i < dstdim; ++i)
				{
					for (int j = 0; j < dstdim; ++j)
					{
						float val = data_raw[sr0 * c + (i + yoffset) * sr1 + (j + xoffset) * sr2];
						channel_avg[c] += val;
						v_max = max(v_max, val);
						v_min = min(v_min, val);
						data[s0 * c + s1 * i + s2 * j] = val;
					}
				}
				channel_avg[c] /= dstdim * dstdim;
				fprintf(stdout, "Average value for channel %d: %.4f\n", c, channel_avg[c]);
			}
			fprintf(stdout, "Value range of input in [%.4f, %.4f]\n", v_min, v_max);
			fprintf(stdout, "Input dimensions: %lix%lix%li\n", input->size[0], input->size[1], input->size[2]);

			// classification
			THTensor *output = overfeat::fprop(input);
			fprintf(stdout, "Output dimensions: %lix%lix%li\n", output->size[0], output->size[1], output->size[2]);
			if ((output->size[1] != 1) || (output->size[2] != 1))
			{
				cerr << "Can only determine class if the output is 1x1. Reduce input size" << endl;
				exit(0);
			}
			output->nDimension = 1;
			overfeat::soft_max(output, probas);
			vector<pair<string, float> > top_classes = overfeat::get_top_classes(probas, nTopClasses);

			// print output
			const char *format = "%50s | %10.5f\n";
			fprintf(stdout, "%50s | %10s\n", "Class", "Rating");
			for (int i = 0; i < nTopClasses; ++i)
			{
				fprintf(stdout, format, top_classes[i].first.c_str(), top_classes[i].second);
			}

		}
		else
		{// if nTopClasses < 0, we output the features

			// extract features
			THTensor *output = overfeat::fprop(input_raw);

			// print output
			THTensor *features = overfeat::get_output(feature_layer);
			real *data = THTensor_(data)(features);
			long
					sf = features->stride[0],
					sy = features->stride[1],
					sx = features->stride[2];
			cout << features->size[0] << " " << features->size[1]
				 << " " << features->size[2] << endl;
			for (int i = 0; i < features->size[0]; ++i)
				for (int y = 0; y < features->size[1]; ++y)
					for (int x = 0; x < features->size[2]; ++x)
						cout << data[i * sf + y * sy + x * sx] << " ";
			cout << endl;
		}
	}

	THTensor_(free)(probas);
	THTensor_(free)(input);
	THTensor_(free)(input_raw);
	overfeat::free();
	return 0;
}
