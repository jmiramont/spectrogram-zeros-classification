clear all; close all;

save_figures = false;

rng(0);
N = 2^9;
Nchirp = N-100;
tmin = round((N-Nchirp)/2);
tmax = tmin + Nchirp;
x = zeros(N,1);
tchirp = (0:Nchirp-1);
instf = 0.1+0.25*tchirp/Nchirp + 0.1*sin(2*pi*tchirp/Nchirp);
xsub = cos(2*pi*cumsum(instf)).'.*tukeywin(Nchirp,0.25);
x(tmin+1:tmax) = xsub;

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.


[F,~,~] = tfrstft(x,1:N,Nfft,w,0);
F = F(1:floor(Nfft*fmax),:);
F = flipud(F);
S = abs(F).^2;


% Noise realization:
SNRin = 15;
noise = randn(size(x));
noise = noise*sqrt(10^(-SNRin/10)*sum(x.^2)/N);
xnoise = x+noise;
gamma = var(noise);
[Fnoise,~,~] = tfrstft(noise,1:N,Nfft,w,0);
Fnoise = Fnoise(1:floor(Nfft*fmax),:);
Fnoise = flipud(Fnoise);
Snoise = abs(Fnoise).^2;

Fmix = F+Fnoise;
Smix = abs(Fmix).^2;

%%
figure()
imagesc(-log(abs(F))); hold on;
contour(S,[gamma gamma],'--g')
title('Signal','Interpreter','latex');
colormap pink
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
axis square;

if save_figures
print_figure('figures/level_set_signal.pdf',4,4,'RemoveMargin',true)
end
%%
figure()
% subplot(1,3,2)
imagesc(-log(abs(Fnoise)));
ceros1 = find_zeros_stft(Snoise); hold on;
plot(ceros1(:,2),ceros1(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',1.0);
title('Noise Only','Interpreter','latex');
colormap pink
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
axis square;

if save_figures
print_figure('figures/level_set_noise.pdf',4,4,'RemoveMargin',true)
end
%%
% figure()
% % subplot(1,3,3)
% imagesc(-log(abs(Fmix))); hold on;
% ceros2 = find_zeros_stft(Smix);
% plot(ceros2(:,2),ceros2(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',1.0);
% contour(S,[gamma gamma],'--g')
% title('Noise + Signal','Interpreter','latex');
% colormap pink
% xticklabels([]); yticklabels([])
% xticks([]); yticks([])
% xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% axis square;
% 
% if save_figures
% print_figure('figures/level_set_mix.pdf',4,4,'RemoveMargin',true)
% end