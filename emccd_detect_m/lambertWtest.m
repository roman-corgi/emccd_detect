close all
clear
clc
format compact
p2 = @(x,g) (x * exp(-x/g)/g^2)
c2 = @(x,g) (1-(1+x/g)*exp(-x/g))
g2 = @(x,g) (-g*lambertw(-1, (x-1)/exp(1)) - g)

g = 100

x = 220

p2(x, g)

c2(x, g)

xset = 0:10:1000;
test = [];
for x = xset
    test = [test, g2(c2(x, g), g)];
end

figure,
plot(xset, test, 'o-')
grid
