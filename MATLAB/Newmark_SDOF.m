clear all
close all
clc
%%
m=1;
k=1000;
c=0.3;
%%
F_amp=5;
freq=8;
%%
time=50;
dt=0.125/25;
time_vec=0:dt:time;
Force=F_amp*sin(2*pi*freq*time_vec);
%%
[V,lembda]=eig(k,m);
Natural_freq_inHz=sqrt(lembda)/(2*pi)       %Natural Frequency in Hz
%%
x0=0;
v0=0;
[x xd xdd]=BetaNewmark3(m,k,c,time,x0,v0,dt,Force);
plot(time_vec,x)