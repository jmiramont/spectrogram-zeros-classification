% Classification of spectrogram zeros. Example with a bat echolocatization
% signal.
%
% Thanks to Curtis Condon, Ken White, and Al Feng of the Beckman Institute 
% of the University of Illinois for the bat data and for permission to use 
% it in this paper.
%--------------------------------------------------------------------------


clear all; close all;
save_figures = false; % If true, save the figures to .pdf.
rng(0)

% Load the signal
load batsig.mat
x = batsig;
x = x - mean(x);
x = x / max(abs(x));

N = length(batsig); tmin=1;tmax = N;

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.
[F,~,~] = tfrstft(x,1:N,Nfft,w,0);
S = abs(F).^2;

% %  Padding the signal to next power of 2 with very low-amplitude noise
% (when using the spectrogram zeros we cannot pad with zeros).
N2 = 2^nextpow2(N);
tmin = ceil((N2-N)/2);
tmax = tmin + N;
xpad = randn(N2,1)*1e-8;
xpad(tmin+1:tmax) = x;
N = N2;
x = xpad;


% Noise realization:
J = 1024;
r = 3*ceil(T/8);

% Filtering using classification zeros:
[mask, signal_r, TRI, TRIselected, ceros, F, class, K, Hceros, zeros_hist] =...
    classified_zeros_denoising(x, 'estimate', J, r, [10,0]);

% Changes for the figures:
F = flipud(F(1:N+1,:));
ceros(:,1) = N +1 - ceros(:,1)  ;
zeros_hist = flipud(zeros_hist);
mask = flipud(mask(1:N+1,:));

%% Show the histogram.
figure()
imagesc(log(zeros_hist))
colormap gray
xticklabels([]); yticklabels([])
xticks([]); yticks([]); xlim([tmin, tmax]);
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% title('2D histogram of zeros','Interpreter','latex')

if save_figures
print_figure('figures/histogram_batsignal.pdf',4,4,'RemoveMargin',true)
end

%% Spectrogram and zeros.
figure()
% subplot(1,2,1)
imagesc((abs(F))); hold on;
plot(ceros(:,2),ceros(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',1.5);
xticklabels([]); yticklabels([])
xticks([]); yticks([]); xlim([tmin, tmax]);
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% title('Log-spectrogram and zeros','Interpreter','latex')
% colormap pink
if save_figures
print_figure('figures/spectrogram_batsignal.pdf',4,4,'RemoveMargin',true)
end

%% Filtering mask
figure()
imagesc(mask)
xticklabels([]); yticklabels([])
xticks([]); yticks([]); xlim([tmin, tmax]);
xlabel('time'); ylabel('frequency')
title('Extraction Mask')
colormap bone
axis square

%% Classified zeros.
figure()
imagesc(-log(abs(F))); hold on;
% triplot(TRIselected,ceros(:,2),ceros(:,1),'c','LineWidth',0.5);
colores = string({'blue';'green';'red'});
symbols = string({'o';'^';'d'});
for i = 1:3
    plot(ceros(class==i,2),...
        ceros(class==i,1),symbols(i),...
        'Color',colores(i),'MarkerFaceColor',colores(i),'MarkerSize',1.5);
end
xticklabels([]); yticklabels([])
xticks([]); yticks([]); xlim([tmin, tmax]);
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% title('Log-Spectrogram and zeros','Interpreter','latex')
colormap pink
axis square

if save_figures
print_figure('figures/batsignal.pdf',4,4,'RemoveMargin',true)
end