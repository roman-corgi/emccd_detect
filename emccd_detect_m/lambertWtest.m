close all
clear 
clc
p2 = @(x,g) (x * exp(-x/g)/g^2)
c2 = @(x,g) (1-(1+x/g)*exp(-x/g))
g2 = @(x,g) (g*lambertW(-1, (x-1)/exp(1)) - g)

g = 100

x = 220

p2(x, g)

c2(x, g)

g2(c2(x, g), g)

