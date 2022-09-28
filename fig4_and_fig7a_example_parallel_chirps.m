% Classification of spectrogram zeros. Example with parallel linear chirps.

clear all; close all;
save_figures = true; % If true, save the figures to .pdf.
rng(0)

% Generate the signal
N = 2^9;
Nchirp = N;
tmin = round((N-Nchirp)/2);
tmax = tmin + Nchirp;
x = zeros(N,1);
instf1 = 0.1+0.15*(0:Nchirp-1)/Nchirp;
instf2 = 0.15+0.15*(0:Nchirp-1)/Nchirp;
% x(tmin+1:tmax) = (fmlin(Nchirp,0.10,0.25) + fmlin(Nchirp,0.15,0.3)).*tukeywin(Nchirp,0.25);
x(tmin+1:tmax) = (cos(2*pi*cumsum(instf1)) + cos(2*pi*cumsum(instf2))).*tukeywin(Nchirp,0.25).';

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

% Noise realization:
J = 512;
dT = 3*ceil(T/8);
SNRin = 30;
noise = randn(size(x));% + 1i*randn(size(x));
[xnoise,std_noise] = sigmerge(x,noise,SNRin);

%% Filtering using classification zeros:
[mask, signal_r, TRI, TRIselected, ceros, F, class, K, Hceros, zeros_hist] =...
    classified_zeros_denoising(xnoise, 'estimate', J, dT, [5 5]);

% Changes for the figures:
F = flipud(F(1:N+1,:));
ceros(:,1) = N +1 - ceros(:,1)  ;
zeros_hist = flipud(zeros_hist);

QRF = 20*log10(norm(x)/norm(x-signal_r));


%% Circles with the patch radius on the histogram:
figure()
imagesc((zeros_hist).^0.3); hold on;
colormap jet
% plot(ceros(:,2),ceros(:,1),'o','Color','r','MarkerFaceColor','r',...
%     'MarkerSize',4); hold on;
viscircles(fliplr(ceros),dT*ones(size(ceros,1),1))
axis square

%% Spectrogram and zeros.
figure()
% subplot(1,2,1)
imagesc(-log(abs(F))); hold on;
plot(ceros(:,2),ceros(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',2);
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% ylim([1 round(Nfft/2+1)])
title('Log-Spectrogram and zeros','Interpreter','latex')
colormap pink

if save_figures
    print_figure('figures/spectrogram_parallel.pdf',7,4,'RemoveMargin',true)
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
imagesc(-log(abs(F).^2)); hold on;
% triplot(TRIselected,ceros(:,2),ceros(:,1),'c','LineWidth',0.5);
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
% title('Log-Spectrogram and zeros','Interpreter','latex')
colormap pink
% axis square

if save_figures
    print_figure('figures/parallel_chirps.pdf',6,4,'RemoveMargin',true)
end

%%
labels = string({'First kind','Second kind','Third kind'});
figure()
for i = 1:3
    X = Hceros(class==i,:);
    C(i,:) = mean(X,1);
    plot(X(:,1),...
         X(:,2),symbols(i),...
        'Color',colores(i),...
        'MarkerFaceColor',colores(i),...
        'MarkerSize',3,...
        'DisplayName',labels(i)); hold on;
end


plot(C(:,1),C(:,2),'o','MarkerFaceColor','c',...
                        'MarkerEdgeColor','m',...
                        'MarkerSize',4,...
                        'DisplayName','Centroids');

grid on;
xlabel('$\Vert B(z,r) \Vert_{1}$ (normalized)','Interpreter','latex');
ylabel('$H_{1}(B(z,r))$ (normalized)','Interpreter','latex');
xticklabels([]); yticklabels([])
xticks([]); yticks([])
legend('boxoff');
legend('Location','southwest');

if save_figures
print_figure('figures/feature_space_parallel.pdf',7,4,'RemoveMargin',true)
end
