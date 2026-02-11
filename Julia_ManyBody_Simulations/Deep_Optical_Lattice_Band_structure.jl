using LinearAlgebra
using Plots


lambda =1 #units measured in wavelength
kl=2*pi/lambda 
a=lambda/2
V0=3 
qs =[-pi/a:0.1:pi/a;]
l=10
js=[-l:l;]


bstr=zeros(length(qs), length(js))

for qq = 1:length(qs)
    q=qs[qq];
    d0=(2 .* js .+ q ./ kl).^2 .+ V0/2
    d1=-V0/4 .*ones(length(d0)-1)


    H=SymTridiagonal(d0,d1)
    bstr[qq,:]=eigvals(H)

end

plot(qs,bstr[:,1:5], xlabel="qa", ylabel="Energy")

display(current())