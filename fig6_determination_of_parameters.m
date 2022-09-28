% Study of parameters of an unsupervised method to classify the zeros of
% the spectrogram. The first section of the script runs the experiment.
% Alternatively, you can run only the second section, provided that the
% file with the results is in the same folder or in the path.
%
% September 2022
% Author: Juan M. Miramont <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------

%% Section 1: Computation.
clear all; close all

% Generate the signal.
N = 2^8;
[x, det_zeros, impulses_location] = triple_tone_signal(N);

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.
dT = [ceil(T/8) ceil(T/4) 3*ceil(T/8) ceil(T/2) 5*ceil(T/8) 6*ceil(T/8)];
% dT = 3*ceil(T/8);
margins = [5, round(T)];

% Number of repetitions of the experiment.
nreps = 100;

% 2D histograms parameters:
M = 512; % Number of realizations.
rng(0);
noise_matrix = randn(nreps,N);
noise_matrix = noise_matrix./std(noise_matrix,[],2);

SNRin = [0 5 10 20 30];
prop = [0.7 0.85 1.0 1.15 1.3];


for k = 1:nreps
    disp(k)
    noise = noise_matrix(k,:);

    for i =1:length(SNRin)
        [xnoise,std_noise] = sigmerge(x,noise.',SNRin(i));

        % Compute STFT and estimate noise std.
        [F,~,~] = tfrstft(xnoise,1:N,Nfft,w,0);

        [zeros_hist, ceros, F, ~, ~, ~, S] = ...
            compute_zeros_histogram(xnoise, 'estimate', M, margins);

        % Determine the true class of the zeros:
        true_class = ...
            determine_true_class_2(ceros,det_zeros,x,var(std_noise*noise),margins);

        for p = 1:length(dT)

            % Classify the zeros of the spectrogram:
            zeros_hist = imgaussfilt(zeros_hist);
            [class, K(i,k,p)] = classify_spectrogram_zeros(zeros_hist, ceros, dT(p));
            %             class = classify_spectrogram_zeros(zeros_hist, ceros, 9);

            if any(isnan(class))
                accuracy(i,k,p)= nan;
            else
                accuracy(i,k,p) = sum(class==true_class)/length(class);
            end
 

%                             figure()
%                             colores = string({'blue';'green';'red';});
%                             symbols = string({'o';'^';'d'});
%             
%                             subplot(1,2,1)
%                             imagesc(-log(S)); hold on;
%                             for o = 1:3
%                                 plot(ceros(true_class==o,2),ceros(true_class==o,1),symbols(o),...
%                                     'Color',colores(o),'MarkerFaceColor',colores(o),'MarkerSize',2);
%                             end
%                             xticklabels([]); yticklabels([])
%                             xticks([]); yticks([])
%                             xlabel('time'); ylabel('frequency')
%                             title('True Class')
%                             colormap pink
%             
%                             subplot(1,2,2)
%                             imagesc(-log(S)); hold on;
%                             for o = 1:3
%                                 plot(ceros(class==o,2),ceros(class==o,1),symbols(o),...
%                                     'Color',colores(o),'MarkerFaceColor',colores(o),'MarkerSize',2);
%                             end
%                             xticklabels([]); yticklabels([])
%                             xticks([]); yticks([])
%                             xlabel('time'); ylabel('frequency')
%                             title('Predicted Class')
%                             colormap pink

        end
    end
end

save params_K_100_M_512_beta_TONES_KMEANS_shannon.mat

%% Section 2: Results.
% Run this section to generate the figure with the summary of the results.
% The file params_K_100_M_512_beta_TONES_KMEANS_shannon.mat must be in the
% path or in the same folder as this script.

close all; clear all;
load params_K_100_M_512_beta_TONES_KMEANS_shannon.mat;

figure()
aux = squeeze(accuracy(:,:,1:5));
center = squeeze(median(aux,2,'omitnan'))*100;
yneg = squeeze(quantile(aux,0.25,2))*100-center;
ypos = squeeze(quantile(aux,0.75,2))*100-center;

b = bar(center.','EdgeColor','none');

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(center.');
% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
% errorbar(x',center.',errorstd.','k','linestyle','none');
errorbar(x',center.',yneg.',ypos.','k','linestyle','none','CapSize',1);
hold off

ylabel('Acc$(r, J=512)$ (\%)','Interpreter','latex' );
ylim([0 105])
yticks(0:20:100);
xticklabels({'$T/8$', '$T/4$', '$3T/8$', '$T/2$', '$5T/8$'});
xlabel('$r$','FontSize',10, 'Interpreter','latex')
%     title('$\operatorname{SNR}(x,\xi) = ' + string(SNRin(j))+ ' dB$');
% title('SNR$(x,\xi)$ = ' + string(SNRin(j))+ ' dB','Interpreter','latex');
grid on

leg = legend({  'SNR$(x,\xi) = 0$ dB',...
                'SNR$(x,\xi) = 5$ dB',...
                'SNR$(x,\xi) = 10$ dB',...
                'SNR$(x,\xi) = 20$ dB',...
                'SNR$(x,\xi) = 30$ dB'},'Location','northoutside','FontSize',7,...
    'Interpreter','latex');
leg.ItemTokenSize = [4,4];
leg.NumColumns = 3;
legend('boxoff')

xaxisproperties= get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex';
xaxisproperties.TickLabelInterpreter = 'latex';

% print_figure('parameters_bars_K_100_db.pdf',9,5,'RemoveMargin',false);
