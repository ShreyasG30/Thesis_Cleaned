clear all
close all
clc

K1=4000; K2=2000; K3=6000;
M1=1; M2=2;

M=[M1 0;
    0 M2];
K=[K1+K2 -K2;
    -K2 K2+K3];

w_f1=6; F1=20;
w_f2=0; F2=0;

[modeShape freqz]=eig(K,M);

A00=zeros(2); A11=eye(2);
CC=[A00 A11;-inv(M)*K A00];
global CC F1 F2 w_f1 w_f2 M

max_freq=max(sqrt(diag(freqz))/(2*pi));
dt=1/(max_freq*20);
time=0:dt:500*dt;

y0=[0 0 0 0];
[tsol,ysol]=ode23('testodeF_2D',time,y0);
plot(time,ysol(:,1:2),'linewidth',2)
xlabel('Time(sec)')
ylabel('displacement(m)')
grid on
