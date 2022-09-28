% 2D histograms of the spectrogram zeros with different amplitudes of
% noise.

clear all; close all;
% Seed the random number generator.
rng(0)

% Generate the signal.
N = 2^8;
Nchirp = N;
tmin = round((N-Nchirp)/2);
tmax = tmin + Nchirp;
x = zeros(N,1);
instf1 = 0.1+0.2*(0:Nchirp-1)/Nchirp;
instf2 = 0.15+0.2*(0:Nchirp-1)/Nchirp;
x(tmin+1:tmax) = (cos(2*pi*cumsum(instf1)) + cos(2*pi*cumsum(instf2))).*tukeywin(Nchirp,0.25).';

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

% Signal + noise mixture.
SNRin = 30;
original_noise = randn(size(x));
[xnoise,std_noise] = sigmerge(x,original_noise,SNRin);
[F,~,~] = tfrstft(xnoise,1:N,Nfft,w,0);
F = F(1:floor(Nfft*fmax),:);
F = flipud(F);
S = abs(F).^2;

% Find original zeros.
ceros = find_zeros_stft(S);

% Keep zeros within margins:
margin_row = 5; margin_col = 5;
invalid_ceros = zeros(length(ceros),1);
invalid_ceros(ceros(:,1)<margin_row | ceros(:,1)>(size(S,1)-margin_row))=1;
invalid_ceros(ceros(:,2)<margin_col | ceros(:,2)>(size(S,2)-margin_col))=1;
invalid_ceros = logical(invalid_ceros);
valid_ceros = ~invalid_ceros;
% number_of_valid_ceros = sum(valid_ceros);
ceros=ceros(valid_ceros,:);

% figure()
imagesc(-log(abs(F))); hold on;
plot(ceros(:,2),ceros(:,1),'wo','MarkerFaceColor','w','MarkerSize',0.75);
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('Log-Spectrogram','Interpreter','latex')
colormap pink
% print_figure('figures/spectrogram_a.pdf',5,4,'RemoveMargin',true)

%% Histogram using noise with the same variance as the original.
disp('Computing histogram...');
lims = 0;
aux_plane = zeros(size(S));
J = 2048; % Number of noise realizations.
% tic()
parfor j  = 1:J
    noise_alg = randn(N,1); 
    noise_alg = noise_alg/std(noise_alg)*std_noise;
    xnoise_alg = xnoise+noise_alg;
%     xnoise_alg = sigmerge(xnoise,noise_alg,SNRalg);
    [S_alg,~,~] = tfrsp(xnoise_alg,1:N,Nfft,w,0);
    S_alg = S_alg(1:floor(Nfft*fmax),:);
    S_alg = flipud(S_alg);
    [~, Qz] = find_zeros_stft(S_alg);
    aux_plane = aux_plane + Qz;
end
% toc()
hist2d = aux_plane(lims+1:end-lims,lims+1:end-lims);
selected_hist = hist2d;

disp('Finished.');

figure()
imagesc(log(selected_hist))
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('2D Histogram of Zeros','Interpreter','latex')
colormap gray
% print_figure('figures/histogram_zeros_just.pdf',6,4,'RemoveMargin',true)


%% Histogram using noise with half the variance of the original noise
disp('Computing histogram...');
lims = 0;
aux_plane = zeros(size(S));

parfor j  = 1:J
    noise_alg = randn(N,1); 
    noise_alg = noise_alg/std(noise_alg)*sqrt(0.05)*std_noise;
    xnoise_alg = xnoise+noise_alg;
%     xnoise_alg = sigmerge(xnoise,noise_alg,SNRalg);
    [S_alg,~,~] = tfrsp(xnoise_alg,1:N,Nfft,w,0);
    S_alg = S_alg(1:floor(Nfft*fmax),:);
    S_alg = flipud(S_alg);
    [~, Qz] = find_zeros_stft(S_alg);
    aux_plane = aux_plane + Qz;
end

hist2d = aux_plane(lims+1:end-lims,lims+1:end-lims);
selected_hist = hist2d;
disp('Finished.');

figure()
imagesc(log(selected_hist))
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('2D Histogram of Zeros','Interpreter','latex')
colormap gray
% print_figure('figures/histogram_zeros_noiseless.pdf',6,4,'RemoveMargin',true)

%% Histogram using noise with two times the variance of the original noise

disp('Computing histogram...');
lims = 0;
aux_plane = zeros(size(S));

parfor j  = 1:J
    noise_alg = randn(N,1); 
    noise_alg = noise_alg/std(noise_alg)*sqrt(20)*std_noise;
    xnoise_alg = xnoise+noise_alg;
    [S_alg,~,~] = tfrsp(xnoise_alg,1:N,Nfft,w,0);
    S_alg = S_alg(1:floor(Nfft*fmax),:);
    S_alg = flipud(S_alg);
    [~, Qz] = find_zeros_stft(S_alg);
    aux_plane = aux_plane + Qz;
end

hist2d = aux_plane(lims+1:end-lims,lims+1:end-lims);
selected_hist = hist2d;
disp('Finished.');

%
figure()
imagesc(log(selected_hist))
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('2D Histogram of Zeros','Interpreter','latex')
colormap gray
% print_figure('figures/histogram_zeros_noisy.pdf',6,4,'RemoveMargin',true)