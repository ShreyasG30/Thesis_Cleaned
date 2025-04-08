clear all
close all
clc
%%
m=1;
k=20;
c=0.2 ;
[a,b] = eig(k,m);
Natural_frw_inHz=sqrt(b)/(2*pi)
%%
x0=0.01;
v0=0;
%%
time=50;
dt=0.01;

[x xd xdd]=BetaNewmark2(m,k,c,time,x0,v0,dt);
time_vec=0:dt:time;
plot(time_vec,x)