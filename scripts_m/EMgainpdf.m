function p = EMgainpdf (x, n, g)



p = x.^(n-1) .* exp(-x/g) / (g^n * factorial(n-1))  ;  


end