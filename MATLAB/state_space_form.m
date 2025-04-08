clear all
close all
clc

global m k  % Declare global variables in the script before calling the function
m = 2;      % Example mass value (modify as needed)
k = 2000;     % Example stiffness value (modify as needed)

y0 = [0.05; 0];   % Initial conditions: y(0) = 1, dy/dt(0) = 0
tspan = 0:0.002:2;  % Time range

[tsol, ysol] = ode45(@testode1, tspan, y0) % Use function handle @testode1

% Plot results
figure;
plot(tsol, ysol(:,1), 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Displacement');
title('Mass-Spring System Response');
grid on;
