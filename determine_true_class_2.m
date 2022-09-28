function true_class = determine_true_class_2(zeros_pos,det_zeros,x,gamma,margins)
% Determine the true class of the given zeros using the signal and the
% known variance gamma^2 of the noise.
% The zeros closer to the deterministic zeros are labeled as zeros of the
% first kind,
% The zeros closer to the level curve given by the S = gamma^2 are labeled
% as zeros of the third kind.
% The remaining zeros are considered zeros of the second kind.
% This classification is later used as a ground-truth to estimate the
% performance of an unsupervised classification of the zeros.
%
% Input:
% - zeros_pos:  Zeros of the spectrogram.
% - det_zeros:  Deterministic zeros computed by the known properties of the
%               three parallel tones signal.
% - x:          A signal comprising three parallel tones obtained with the
%               function triple_tone_signal() function.
% - gamma:      Standard deviation of the noise realization used.
% - margins:    Margins to reduce border effects.
%
% Output:
% - true_class: The corresponding kind of zero:
%               .First kind: Interference signal-signal.
%               .Second kind: Interference noise-noise.
%               .Third kind: Interference signal-noise.
%
% September 2022
% Author: Juan M. Miramont <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------

N = length(x);
Nfft = 2*N;
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.
T = round(T);
[S,~,~] = tfrsp(x,1:N,Nfft,w,0);
S = S(1:round(Nfft/2+1),:);
C = contourc(S,[gamma gamma]);
C = round(fliplr(C.'));
C(C(:,1)<margins(1) | C(:,1)>N-margins(1),:) = [];
C(C(:,2)<margins(2) | C(:,2)>N-margins(2),:) = [];

% figure()
% imagesc(S); hold on;
% plot(round(C(:,2)),round(C(:,1)),'g.');

% colormap pink
% det_zeros = [det_zeros(:,2) det_zeros(:,1)];
det_zeros = fliplr(det_zeros);
% det_zeros(det_zeros(:,1)<76 | det_zeros(:,1)>180,:) = [];
true_class = zeros(size(zeros_pos,1),1);

% Create a KD tree to search neighboring points to the zeros.
Mdl = KDTreeSearcher(zeros_pos);

% Search for zeros closer to the positions of the impulses:
zeros_near_impulses = knnsearch(Mdl,C,'K',1);
true_class(unique(zeros_near_impulses)) = 3;

% Search for deterministic zeros
[deterministic_zeros, distance] = knnsearch(Mdl,det_zeros,'K',1);
% deterministic_zeros(distance>2*sqrt(2)) = [];
true_class(unique(deterministic_zeros)) = 1;
true_class(true_class==0) = 2;

