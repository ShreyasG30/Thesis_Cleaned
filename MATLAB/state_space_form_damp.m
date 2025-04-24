clc;
clear all;
close all;

K1=4000; K2=200; K3=6000;
M1=1; M2=2;
C1=1; C2=2; C3=0;

M=[M1 0;
    0 M2];
K=[K1+K2 -K2;
    K2 K2+K3];
C=[C1+C2 -C2;
    -C2 C2+C3];

[modeShape, fr]=eig(K,M);

A1=zeros(2); A2=eye(2);
CC=[A1 A2;-inv(M)*K -inv(M)*C];
global CC

max_freq=max(sqrt(diag(fr))/(2*pi));
dt=1/(max_freq*20);
time=0:dt:500*dt;

y0=[0.01 0 0 0];
[tsol, ysol]=ode23('testode_2D',time,y0);
plot(time,ysol(:,1),'linewidth',2)
xlabel('Time(sec)')
ylabel('displacement(m)')
ylim([-.02 .02])
grid on