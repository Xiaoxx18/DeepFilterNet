#include <iostream>
#include <vector>


void band_mean_norm_erb(std::vector<float>& input, std::vector<float>& state, float alpha = 0.99) {
	for (int i = 0; i < input.size(); ++i) {
		state[i] = input[i] * (1. - alpha) + state[i] * alpha;
		input[i] = (input[i] - state[i]) / 40.0;
	}
}

void erb_norm(std::vector<std::vector<float>>& input, float alpha = 0.99) {
	float step = (-90 - (-60)) / (input[0].size() - 1);
	std::vector<float> state(32);
	for (int i = 0; i < state.size(); ++i) {
		state[i] = -60 + (i * step);
	}

	for (int i = 0; i < input.size(); ++i) {
		band_mean_norm_erb(input[i], state);
	}
}


void band_unit_norm(std::vector<float>& xs_real, std::vector<float>& xs_imag, std::vector<float>& state, float alpha = 0.99) {
	float mag_abs;
	for (int i = 0; i < xs_real.size(); ++i) {
		mag_abs = sqrt(pow(xs_real[i], 2) + pow(xs_imag[i], 2));
		state[i] = sqrt(mag_abs * (1. - alpha) + state[i] * alpha);
		xs_real[i] = xs_real[i] / state[i];
		xs_imag[i] = xs_imag[i] / state[i];
	}
}

void unit_norm(std::vector<std::vector<float>>& input_real, std::vector<std::vector<float>>& input_imag, float alpha = 0.99) {
	float step = (0.001 - 0.0001) / (input_real[0].size() - 1);
	std::vector<float> state(input_real[0].size());
	for (int i = 0; i < state.size(); ++i) {
		state[i] = 0.001 + (i * step);
	}
	for (int i = 0; i < input_real.size(); ++i) {
		band_unit_norm(input_real[i], input_imag[i], state);
	}
}


void main() {
	/*
	此处调用进行使用即可，也可用于端侧部署DeepFilterNet使用
	*/
}