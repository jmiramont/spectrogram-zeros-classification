function [class,K,features] = classify_spectrogram_zeros(zeros_hist, zeros_pos, r, plot_figures)
% Classify the zeros of the spectogram of a signal given the 2d histogram
% of zeros, computed using compute_zeros_histogram() function.
%
% Other functions needed:
% - compute_centroids()
%
% Input:
% - zeros_hist: Histogram of zeros computed using the function:
%               'compute_zeros_histogram()'.
% - zeros_pos:  A [N,2] array with the time-frequency coordenates of the
%               zeros of the spectrogram. Where N is the number of zeros.
% - r:          Radius of the balls centered at each original zero, where
%               the descriptors are computed.
% - plot_figures: If true, plots the feature space. (Default false).
%
% Output:
% - class:      A [N,1] vector with assigned kind of zeros (1,2 or 3).
% - K:          Number of clusters detected. K=1 means only noise. K=2
%               means a signal is present. K=3 means that zeros of
%               interference between components are present.
% - features:   A [N,2] array with the values of the features computed
%               for each zero.
%
% Example:
%      N = 2^9;
%      x = real(fmlin(N,0.10,0.25)+fmlin(N,0.15,0.3)).*tukeywin(N,0.1);
%      xn = sigmerge(x,randn(size(x)),20);
%      [zeros_hist, zeros_pos, F, w, T, N, S] =...
%                            compute_zeros_histogram(xn, 'estimate');
%      [class,K]  = classify_spectrogram_zeros(zeros_hist, ...
%                                                 zeros_pos, 3*round(T/8));
%      zeros_hist = flipud(zeros_hist);S = flipud(S);
%      zeros_pos(:,1) = N + 1 - zeros_pos(:,1);
%      colores = string({'b';'g';'r'}); symbols = string({'o';'^';'d'});
%      figure(); imagesc(-log(abs(S).^2)); hold on;        
%      for i = 1:K
%        plot(zeros_pos(class==i,2),zeros_pos(class==i,1),symbols(i),...
%        'Color',colores(i),'MarkerFaceColor',colores(i),'MarkerSize',2,...
%        'DisplayName', 'Kind '+string(i));
%      end
%      title('Spectrogram and classified zeros'); legend();
%      xticklabels([]); yticklabels([]); xticks([]); yticks([]);
%      xlabel('time'); ylabel('frequency'); colormap pink;
%
% September 2022
% Author: Juan M. Miramont <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------


if nargin<4
    plot_figures = false;
end


% Compute a KD Tree for easily searching balls of neighbors:
[row,col] = ind2sub(size(zeros_hist),1:numel(zeros_hist));
idx = [row.' col.'];
Mdl = KDTreeSearcher(idx);


% Compute the descriptors for each zero.
for i = 1:size(zeros_pos,1)

    % Circular Patches:
    [patch_ind,D] = rangesearch(Mdl,zeros_pos(i,:),r);
    patch_ind = patch_ind{1};
    patch = zeros_hist(patch_ind);

    % You can try squared patches instead:
    %     coordx = max([ceros(i,2)-dT,1]):...
    %                           min([ceros(i,2)+dT,size(selected_hist,2)]);
    %     coordy = max([ceros(i,1)-dT,1]):...
    %                           min([ceros(i,1)+dT,size(selected_hist,1)]);
    %     patch = selected_hist(coordy,coordx);

    % % Uncomment this to see the circular patches:
    %         aux = zeros(size(zeros_hist));
    %         aux(patch_ind) = 1;
    %         imagesc(aux); hold on;
    %         plot(ceros(:,2),ceros(:,1),'o','Color','r',...
    %                       'MarkerFaceColor','r','MarkerSize',4); hold on;
    %         clf; -> Set a break point in this line to stop the loop.

    % Local Density measures:
    norm_1(i,1) = (norm(patch(:),1));
    %     norm_0(i,1) = sum(patch(:)>0)/length(patch(:)); % ps-norm p=0
    %     max_patch(i,1) = max(patch)/norm(patch(:),1);
    %     mean_patch(i,1) = mean(patch(patch>0))/sum(patch);
    %     cvar(i,1) = std(patch(patch>0))/mean(patch(patch>0));
    %     cvar2(i,1) = iqr(patch(patch>0));

    % Local Concentration measures:
    S = sum(patch(:));  % Normalize the sum within the neighboor.
    entropy_shannon(i,1) = -sum(log2((patch(:)/S)+eps).*(patch(:)/S));
    %     entropy_min(i,1) = -log2(max(patch(:)/S)); % MinEntropy
    %     entropy_collision(i,1) = -log(sum((patch(:)/S+eps).^2));
    %     mean_distance(i,1) = sum(D{1}.*patch)/S;

end

features = [norm_1 entropy_shannon];
features = normalize(features);

cluster_fun = @(DATA,K) kmeans(DATA,K,'Replicates',31,'Distance','cityblock');
eva = evalclusters(features,cluster_fun,'gap','KList',[1,2,3], 'B', 31, 'Distance','cityblock');%,'ReferenceDistribution','uniform');


% cluster_fun = @(DATA,K) clusterdata(DATA,'Distance','cityblock','Maxclust', K);
% cluster_fun = @(DATA,K) clusterdata(DATA,'Linkage','ward','Maxclust',K);
% cluster_fun = @(DATA,K) kmedoids(Hceros,K,'Replicates',31,'Distance','sqeuclidean');
% cluster_fun = @(DATA,K) cluster(fitgmdist(DATA,K,'Replicates',5,...
%                                           'CovarianceType','full',...
%                                           'RegularizationValue',0.01),...
%                                           DATA);
% cluster_fun = @(DATA,K) kmeans(Hceros,K,'Replicates',101,'Distance','sqeuclidean');

% eva = evalclusters(Hceros,cluster_fun,'CalinskiHarabasz','KList',[2,3]);
% eva = evalclusters(Hceros,cluster_fun,'DaviesBouldin','KList',[2,3]);
% eva = evalclusters(Hceros,"linkage",'gap','KList',[1,2,3], 'B', 31,'Distance','sqEuclidean'); %,'ReferenceDistribution','uniform');
% eva = evalclusters(Hceros,'gmdistribution','gap','KList',[2,3]);
% save('eva.mat','eva');

K = eva.OptimalK;
% [~,K] = max(eva.CriterionValues);

if isnan(K)
    nanflag = true;
    disp('Found NaN');
end

class = eva.OptimalY;

[~,C] = compute_centroids(features,class);

% Labeling
if K==1
    class(:) = 2;
end

if K==3

    % Sorting clusters by ascending norm:
    [~,sorted_cluster_norm] = sort(C(:,1),'ascend');
    cluster_lowest_norm = sorted_cluster_norm(1);
    aux_cluster = C(cluster_lowest_norm,:); % Save it for later.
    C(cluster_lowest_norm,:) = inf;

    % Sorting clusters in ascending entropy:
    [~,sorted_cluster_entropy] = sort(C(:,2),'ascend');
    cluster_lowest_entropy = sorted_cluster_entropy(1);
    %         C(sorted_cluster_entropy(1),:) = inf;

    %         labels = string({'Noise-Noise','Noise-Signal','Signal-Signal'});

    % Lowest norm -> Class 2 zeros.
    class(class==cluster_lowest_norm) = 20;

    % Lowest entropy from the remaining clusters -> Class 1 zeros.
    class(class==cluster_lowest_entropy) = 10;

    % The remaining cluster -> Class 3.
    class(class==sorted_cluster_entropy(2)) = 30;

    class = class/10;

    C(cluster_lowest_norm,:) = aux_cluster; % Restore the cluster vals.

end

if K==2
    [~,sorted_sum_patch] = sort(C(:,1),'ascend');
    labels = string({'Noise-Noise','Noise-Signal'});
    class(class==sorted_sum_patch(2)) = 30;
    class(class==sorted_sum_patch(1)) = 20;
    class = class/10;
end


if plot_figures
    figure()
    for i = unique(class).'
        plot(features(class==i,1),...
            features(class==i,2),'o'); hold on;
    end


    plot(C(:,1),C(:,2),'o','MarkerFaceColor','c',...
        'MarkerEdgeColor','m',...
        'MarkerSize',4,...
        'DisplayName','Centroids');


    grid on;
    xlabel('$\Vert G_{z} \Vert_{1}$','Interpreter','latex');
    ylabel('$H_{\infty}(G_{z})$','Interpreter','latex');
    % xticklabels([]); yticklabels([])
    % xticks([]); yticks([])
    legend('boxoff');
    legend('Location','southwest');

end
