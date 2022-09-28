%% Example of spectrogram zeros classification using a signal with three tones. 

clear all; close all;
%% Parallel tones
save_figures = false;

% Signal Length:
N = 2^8;
rng(0);
[x, det_zeros, impulses_location] = triple_tone_signal(N);

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

% Noise realization:
J = 512;
SNRin = 30;
noise = randn(size(x));%+1i*randn(size(x));

xnoise = sigmerge(x,noise,SNRin);

dT =3*ceil(T/8);
% Filtering using classification zeros:
[mask, signal_r, TRI, TRIselected, ceros, F, class, K, Hceros, zeros_hist] =...
    classified_zeros_denoising(xnoise, 'estimate', J, dT, [5,5]);

QRF = 20*log10(norm(x)/norm(x-signal_r));

% Changes for the figures:
F = flipud(F(1:N+1,:));
ceros(:,1) = N +1 - ceros(:,1)  ;
zeros_hist = flipud(zeros_hist);
mask = flipud(mask(1:N+1,:));

%% Circles with the patch radius on the histogram:
figure()
imagesc(zeros_hist.^0.3); hold on;
colormap jet
plot(ceros(:,2),ceros(:,1),'o','Color','r','MarkerFaceColor','r',...
    'MarkerSize',4); hold on;
% viscircles(fliplr(ceros),dT*ones(size(ceros,1),1));
axis square


%% Spectrogram and zeros.
figure()
% subplot(1,2,1)
imagesc(-log(abs(F))); hold on;
plot(ceros(:,2),ceros(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',2);
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('Selected Triangles','Interpreter','latex')
ylim([1 round(Nfft/2+1)])
colormap pink

if save_figures
    print_figure('figures/spectrogram_tones.pdf',7,5,'RemoveMargin',true)
end

%% Filtering mask
figure()
imagesc(mask)
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('Extraction Mask','Interpreter','latex')
colormap bone
axis square

%% Classified zeros.
figure()
imagesc(-log(abs(F))); hold on;
triplot(TRIselected,ceros(:,2),ceros(:,1),'c','LineWidth',0.5);
colores = string({'blue';'green';'red'});
symbols = string({'o';'^';'d'});
for i = 1:3
    plot(ceros(class==i,2),...
        ceros(class==i,1),symbols(i),...
        'Color',colores(i),'MarkerFaceColor',colores(i),'MarkerSize',2);
end
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('Selected Triangles','Interpreter','latex')
colormap pink
% axis square
xlim([round(T) N-round(T)])

if save_figures
    print_figure('figures/parallel_tones_tri.pdf',5.5,4,'RemoveMargin',true)
end

