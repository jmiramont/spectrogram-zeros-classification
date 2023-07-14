%% Setting parameters.
% Run this section to generate the figure with the summary of the results.
% This file was run on Matlab R2022b --------------------------------------

clear all;
% Load a Set of Results
load params_N_512_J_256_GMM_GAP.mat;

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

ylabel('Acc$(\beta, J=128)$ (\%)','Interpreter','latex','FontSize',6);
ylim([0 105])
yticks(0:20:100);
% xticklabels({'$T/8$', '$T/4$', '$3T/8$', '$T/2$', '$5T/8$'});
xticklabels(prop);
xlabel('$\beta$','FontSize',10, 'Interpreter','latex')
%     title('$\operatorname{SNR}(x,\xi) = ' + string(SNRin(j))+ ' dB$');
% title('SNR$(x,\xi)$ = ' + string(SNRin(j))+ ' dB','Interpreter','latex');
grid on

% leg = legend({  'SNR$(x,\xi) = 0$ dB',...
%     'SNR$(x,\xi) = 10$ dB',...
%     'SNR$(x,\xi) = 20$ dB',...
%     'SNR$(x,\xi) = 30$ dB'},'Location','northoutside','FontSize',5,...
%     'Interpreter','latex');
% leg.ItemTokenSize = [4,4];
% leg.NumColumns = 4;
% leg.Position = [0.23,0.95,0.58,0.03];
% legend('boxoff')
% 
% xaxisproperties= get(gca, 'XAxis');
% xaxisproperties.TickLabelInterpreter = 'latex';
% xaxisproperties.TickLabelInterpreter = 'latex';

% print_figure('parameters_bars_J_128_GMM.pdf',8.3,2.75,'RemoveMargin',false, 'FontSize',6);
