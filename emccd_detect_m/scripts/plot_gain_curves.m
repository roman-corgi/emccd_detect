clear;  format compact; close all; clc;
addpath('../util');

gain = 100;
n_array = 1:6;
x_array = 0:10:2000;
pdf = zeros(length(n_array), length(x_array));
legend_str = {};
for n = n_array
    pdf(n, :) = em_gain_pdf(x_array, n, gain);
    legend_str{end+1} = ['n = ', num2str(n)];  %#ok<SAGROW>
end

figure;
plot(x_array/gain, pdf);
ylabel('probability density');
xlabel('output counts / gain');
legend(legend_str);

function pdf = em_gain_pdf(x, n, g)
pdf = x.^(n-1) .* exp(-x/g) / (g^n * factorial(n-1));
end
