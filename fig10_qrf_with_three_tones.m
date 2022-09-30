% Study of the performance of a denosing strategy, based on an unsupervised
% method to classify the zeros of the spectrogram. The first section of
% the script runs the experiment. Alternatively, you can run only the
% second section, provided that the file with the results is in the same
% folder or in the path.
%
% September 2022
% Author: Juan M. Miramont <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------

%% Section 1. Simulations:
clear all;
% Signal Length:
N = 2^8;

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

% Generate a signal with three parallel tones.
[x, det_zeros, impulses_location] = triple_tone_signal(N);

% Noise realizations:
J = 512;
K = 200;

% Save the noise realizations for later.
rng(0);
noise_matrix = randn(K,N);
noise_matrix = noise_matrix./std(noise_matrix,[],2);

% Triangles with an edge longer than lmax are selected, according to the
% algorithm proposed by P. Flandrin in " Time-frequency filtering based on
% spectrogram zeros", IEEE Signal Procesing Letters, 2015.
lmax = 1.3:0.1:1.6;

% Simulate different SNRs.
SNRs = 0:5:30;

for q = 1:length(SNRs)
    disp(q);
    SNRin = SNRs(q);
    for k = 1:K
        noise = noise_matrix(k,:);
        [xnoise,std_noise] = sigmerge(x,noise.',SNRin);
        r = 3*ceil(T/8);

        % Filtering using classification zeros:
        [mask_na, signal_r, TRI, TRIselected, ceros, F, class, nclust(q,k)] =...
            classified_zeros_denoising(xnoise, 'estimate', J, r ,[2,2]);

        % Compute the Quality Reconstruction Factor avoiding border
        % effects:
        Tind = round(1.25*T)+1;
        QRF_noise_assisted(q,k) = 20*log10(norm(x(Tind:end-Tind))/...
            norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind)));
        disp(QRF_noise_assisted);

        %       % Uncomment these lines to compare the ground-truth with the
        %       % results.
        %         figure(); plot(x(Tind:end-Tind)); hold on; plot(signal_r(Tind:end-Tind));
        % %
        %         figure()
        %         % Changes for the figures:
        %         F = flipud(F(1:N+1,:));
        %         ceros(:,1) = N +1 - ceros(:,1)  ;
        %         mask_na = flipud(mask_na(1:N+1,:));
        %
        %
        %
        %         colores = string({'blue';'green';'red';});
        %         symbols = string({'o';'^';'d'});
        %
        %         subplot(1,2,1)
        %         imagesc((abs(F))); hold on;
        %         triplot(TRIselected,ceros(:,2),ceros(:,1),'c');
        %         for i = 1:3
        %             plot(ceros(class==i,2),ceros(class==i,1),symbols(i),...
        %                 'Color',colores(i),'MarkerFaceColor',colores(i),'MarkerSize',2);
        %         end
        %
        %         xticklabels([]); yticklabels([])
        %         xticks([]); yticks([])
        %         xlabel('time'); ylabel('frequency')
        %         title('True Class')
        %         xlim([Tind N-Tind]);
        %         colormap pink
        %         subplot(1,2,2)
        %         imagesc(mask_na)
        %         xlim([Tind N-Tind]);

        for p = 1:length(lmax)

            % Get the spectrogram.
            [F,~,~] = tfrstft(xnoise,1:N,Nfft,w,0);
            F = F(1:floor(Nfft*fmax),:);
            F = flipud(F);
            S = abs(F).^2;

            % Find original zeros
            ceros = find_zeros_stft(S);

            % Keep zeros within margins:
            margin_row = 2; margin_col = 2;
            invalid_ceros = zeros(length(ceros),1);
            invalid_ceros(ceros(:,1)<margin_row | ceros(:,1)>(size(S,1)-margin_row))=1;
            invalid_ceros(ceros(:,2)<margin_col | ceros(:,2)>(size(S,2)-margin_col))=1;
            invalid_ceros = logical(invalid_ceros);
            valid_ceros = ~invalid_ceros;

            % Triangulation of zeros
            u=ceros(:,1);
            v=ceros(:,2);
            TRI = delaunay(u,v);
            TRI2 =  [];
            
            % Keep triangles within the specified margins.
            for j = 1:size(TRI,1)
                if ~any(invalid_ceros(TRI(j,:)))
                    TRI2 = [TRI2; TRI(j,:)];
                end
            end

            % Find edge lengths of all triangles.
            [~,MAX_EDGES,TRI_EDGES] = describe_triangles(TRI2,ceros,Nfft,T);

            % Keep the triangles with an edge larger than lmax
            longTriangulos=zeros(size(TRI2,1),1);
            for i =1:size(TRI2,1)
                if any(TRI_EDGES(i,:)>lmax(p))
                    longTriangulos(i)=1;
                end
            end
            TRIselected=TRI2(logical(longTriangulos),:);

            % Get a 0/1 mask based on the selected triangles.
            mask = mask_from_triangles(F,TRIselected,ceros);

            % Reconstruction and QRF computation.
            signal_r = real(sum(F.*mask))/max(w)/N;
            QRF(q,k,p) = 20*log10(norm(x(Tind:end-Tind))/...
                         norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));
        end
    end
end

% Save results.
% save variables_K_200_LABELING_B_TONES_ESTIMATE_SNR_plain_estimator_nclust_1_2_and_3_shannon.mat

%% Section 2: Load the results and generate the figures.
clear all;
load variables_K_200_LABELING_B_TONES_ESTIMATE_SNR_plain_estimator_nclust_1_2_and_3_shannon.mat

save_figures = false;
figure()
mean_QRF = squeeze(mean(QRF,2));
std_QRF = squeeze(std(QRF,[],2));
colors = {'#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'};
% colors = {'k','b','g','c','b','g','r'};
markers = {'o','+','d','s','*','^','p'};
for i = [1,3]%1:length(LB);% [2,3]
    %     plot(SNRs,mean_QRF(:,i),...
    %         'Marker',markers{i+1},'Color',colors{i+1},...
    %         'MarkerFaceColor',colors{i+1}, 'MarkerSize',2,...
    %         'DisplayName','e_{thr}='+string(LB(i))); hold on;
    % text(10*ones(size(LB)),mean_QRF(3,:),string(LB(i)))

    errorbar(SNRs,...
        mean_QRF(:,i),...
        std_QRF(:,i),...
        'Marker',markers{i+1},'Color',colors{i+1},...
        'MarkerFaceColor',colors{i+1}, 'MarkerSize',1.8,...
        'DisplayName','$\ell_{max}$='+string(lmax(i)),...
        'CapSize',0.6); hold on;
    %             'DisplayName',string(SNRalg(i))+' dB',...

end


% plot(SNRs, mean(QRF_noise_assisted,2),...
%     'Marker',markers{1},'Color',colors{1},...
%     'MarkerFaceColor',colors{1}, 'MarkerSize',2,...
%     'DisplayName','Classif. Zeros');

errorbar(SNRs,...
    mean(QRF_noise_assisted,2),...
    std(QRF_noise_assisted,[],2),...
    'Marker',markers{1},'Color',colors{1},...
    'MarkerFaceColor',colors{1}, 'MarkerSize',1.8,...
    'DisplayName','Classif. Zeros',...
    'CapSize',0.6); hold on;


grid on
xlabel('SNRin (dB)', 'Interpreter','latex'); ylabel('QRF (dB)', 'Interpreter','latex');
xlim([SNRs(1)-2 SNRs(end)+2]);
ylim([-2 36]);

leg = legend('Location','northwest','FontSize',7,'Interpreter','latex');
leg.ItemTokenSize = [10,10];
legend('boxoff')
if save_figures
    print_figure('snr_comparison.pdf',8,5,'RemoveMargin',true)
end