clear all;
close all;
Sweep1 = load('combined_data_5umUndercut.mat');
data = Sweep1.combined_data;

R = data(:,1)./1e3;    %Radius in Microns
T = data(:,2);      %Thickness in nm
FSR = data(:,3)./1e9;       %Frequency in GHz
Overlap = data(:,4);        %Mode Overlap

%%
unique_T = unique(T);
colors = lines(length(unique_T));  % Use a colormap with enough distinct colors

figure;
hold on;

% Plot each group of points with a different color
for i = 1:length(unique_T)
    idx = T == unique_T(i);
    scatter(R(idx), FSR(idx), 100, 'filled', 'MarkerFaceColor', colors(i, :));
end

xlabel('Radius (um)');
ylabel('FSR (GHz)');
box on;
set(gca, 'linewidth', 2);
set(gca, 'FontSize', 18);
grid on;
% Create a legend with the unique thickness values
legend(arrayfun(@(x) sprintf('T = %d', x), unique_T, 'UniformOutput', false));
xlim([8, 52])
grid on;
hold off;
saveas(gcf, 'FSR_vs_R.png');

%%
unique_T = unique(T);
colors = lines(length(unique_T));  % Use a colormap with enough distinct colors

figure;
hold on;

% Plot each group of points with a different color
for i = 1:length(unique_T)
    idx = T == unique_T(i);
    scatter(R(idx), Overlap(idx), 100, 'filled', 'MarkerFaceColor', colors(i, :));
end

xlabel('Radius (um)');
ylabel('Mode Overlap');
box on;
set(gca, 'linewidth', 2);
set(gca, 'FontSize', 18);
grid on;
% Create a legend with the unique thickness values
legend(arrayfun(@(x) sprintf('T = %d', x), unique_T, 'UniformOutput', false), 'NumColumns', 4, 'FontSize', 8, 'location', 'nw');
xlim([8, 52])
grid on;
hold off;
saveas(gcf, 'MO_vs_R.png');


%%
figure;
scatter3(R, T, FSR, 'filled');
xlabel('Radius (um)');
ylabel('Thickness (nm)');
zlabel('FSR (GHz)');
title('FSR vs Radius and Thickness');
grid on;
zlim([0, 1000])
set(gca, 'linewidth', 3.5);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Calibri');
saveas(gcf, 'FSR_vs_R_T.png');

%%
figure;
scatter3(R, T, Overlap, 'filled');
xlabel('Radius (um)');
ylabel('Thickness (nm)');
zlabel('Mode Overlap');
title('Overlap vs Radius and Thickness');
grid on;
% zlim([0, 1000])
set(gca, 'linewidth', 3.5);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Calibri');
saveas(gcf, 'MO_vs_R_T.png');

%%
% Filter the data where delta_f < 100
filter_idx = FSR < 100;
R_filtered = R(filter_idx);
T_filtered = T(filter_idx);
delta_f_filtered = FSR(filter_idx);
mode_overlap_filtered = Overlap(filter_idx);

% Define unique radius and thickness values for color-coding
unique_R = unique(R_filtered);
unique_T = unique(T_filtered);
colors_R = lines(length(unique_R)); % Distinct colors for R
colors_T = lines(length(unique_T)); % Distinct colors for T

% Create a scatter plot
figure;
hold on;

% Color-code and plot points based on R and T
for i = 1:length(unique_R)
    for j = 1:length(unique_T)
        idx = (R_filtered == unique_R(i)) & (T_filtered == unique_T(j));
        if any(idx)
            scatter(delta_f_filtered(idx), mode_overlap_filtered(idx), 100, ...
                'MarkerEdgeColor', colors_R(i, :), 'MarkerFaceColor', colors_T(j, :), 'DisplayName', ...
                sprintf('R = %d, T = %d', unique_R(i), unique_T(j)));
        end
    end
end

xlabel('FSR (GHz)');
ylabel('Mode Overlap');
legend('show');
grid on;
box on;
% zlim([0, 1000])
set(gca, 'linewidth', 3.5);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Calibri');