clear all;  close all;

N = 2^8;
J = 1024;

% Parameters for the STFT.
Nfft = 2*N;
[~,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.
dT = 3*ceil(T/8);
% [x, det_zeros, impulses_location] = triple_tone_signal(N);

Nchirp = N;
tmin = round((N-Nchirp)/2);
tmax = tmin + Nchirp;
x = zeros(N,1);
instf1 = 0.1+0.3*(0:Nchirp-1)/Nchirp;
x(tmin+1:tmax) = (cos(2*pi*cumsum(instf1))).*tukeywin(Nchirp,0.5).';

%%
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.
[F,~,~] = tfrstft(x,1:N,Nfft,w,0);
F = F(1:N+1,:);
F = flipud(F);
S = abs(F).^2;
figure(); imagesc(S);


rng(0)
reps = 50;
noise_matrix = randn(30,reps,N);
% save('noise_matrix.mat','noise_matrix','x');

for p = 1:30
for q = 1:reps
    
    noise = squeeze(noise_matrix(p,q,:));
    
    [~, ~, ~, ~, ~, ~, ~, nclust_noise(q,p)] =...
        noise_assisted_method(noise, 'estimate', J, dT ,[2,2]);
    
    [signal_0, std_noise_0] = sigmerge(x,noise,0);
    [~, ~, ~, ~, ~, ~, ~, nclust_signal_0(q,p)] =...
        noise_assisted_method(signal_0, 'estimate', J, dT ,[2,2]);

    [signal_5, std_noise_5] = sigmerge(x,noise,5);
    [~, ~, ~, ~, ~, ~, ~, nclust_signal_5(q,p)] =...
        noise_assisted_method(signal_5, 'estimate', J, dT ,[2,2]);

    [signal_10, std_noise_10] = sigmerge(x,noise,10);
    [~, ~, ~, ~, ~, ~, ~, nclust_signal_10(q,p)] =...
        noise_assisted_method(signal_10, 'estimate', J, dT ,[2,2]);
    
    %     disp(nclust_signal);
end
end


especificity_mean = mean(sum(nclust_noise==1)/size(nclust_noise,1));
especificity_std = std(sum(nclust_noise==1)/size(nclust_noise,1));

sensitivity_0_mean = mean(sum(nclust_signal_0>1)/size(nclust_signal_0,1));
sensitivity_0_std = std(sum(nclust_signal_0>1)/size(nclust_signal_0,1));

sensitivity_5_mean = mean(sum(nclust_signal_5>1)/size(nclust_signal_5,1));
sensitivity_5_std = std(sum(nclust_signal_5>1)/size(nclust_signal_5,1));

sensitivity_10_mean = mean(sum(nclust_signal_10>1)/size(nclust_signal_10,1));
sensitivity_10_std = std(sum(nclust_signal_10>1)/size(nclust_signal_10,1));


% detection_performance_results_matrix = [nclust_noise nclust_signal_0 nclust_signal_5 nclust_signal_10];
% detection_performance_results_matrix(detection_performance_results_matrix==1) = 0;
% detection_performance_results_matrix(detection_performance_results_matrix>0) = 1;

save results_J_1024_detection_performance_30_samples.mat

%
% %% Spectrogram and zeros.
% figure()
% % subplot(1,2,1)
% imagesc(-log(abs(F))); hold on;
% plot(ceros(:,2),ceros(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',2);
% xticklabels([]); yticklabels([])
% xticks([]); yticks([])
% xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% % ylim([1 round(Nfft/2+1)])
% title('Log-Spectrogram and zeros','Interpreter','latex')
% colormap pink


%% Filtering mask
% figure()
% imagesc(mask)
% xticklabels([]); yticklabels([])
% xticks([]); yticks([])
% xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% title('Extraction Mask','Interpreter','latex')
% colormap bone
% axis square


% clear all; load detection_perf_M_512_filtered.mat
