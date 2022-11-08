clc
clear
n = 5;                   % Decomposition Level
w = 'sym4';              % Near symmetric wavelet
load('datasets/SA/Salinas_corrected.mat')
WT = wavedec3(salinas_corrected, n, w);
A = cell(1, n);
D = cell(1, n);
for k = 1:n
A{k} = waverec3(WT, 'a', k);   % Approximations (low-pass components)
D{k} = waverec3(WT, 'd', k);   % Details (high-pass components)
end

reconstructed = waverec3(WT, 'a', 4);
save('datasets/SA/SA_verylow.mat', 'reconstructed')

reconstructed = waverec3(WT, 'a', 3);
save('datasets/SA/SA_low.mat', 'reconstructed')

reconstructed = waverec3(WT, 'a', 2);
save('datasets/SA/SA_medium.mat', 'reconstructed')

reconstructed = waverec3(WT, 'a', 1);
save('datasets/SA/SA_high.mat', 'reconstructed')