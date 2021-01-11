% peak.m - peak-enhancement, plot peaks as Dirac delta functions
% Marquette University, Milwaukee, WI USA
% Copyright 2003, 2004 - All rights reserved.
% Fred J. Frigo, James A. Heinen
%
% S - Input is 2D (real or complex) surface.
% thresh - threshold; peaks must have magnitude greater than this
% suppreslastrow - if set to 1, ignore peaks from last row (recommended)
% Sp - Output representing 2D peak-enhanced surface
%
function Sp=peak(S,thresh,suppresslastrow)
S=abs(S);
[Nsig,Nw]=size(S);
Sb=zeros(Nsig+2,Nw+2);
Sbp=zeros(Nsig+2,Nw+2);
for m=1:Nsig
    for k=1:Nw
        Sb(m+1,k+1)=S(m,k);
    end
end
for m=2:Nsig+1
    for k=2:Nw+1
        if Sb(m,k)>=max([Sb(m+1,k-1) Sb(m+1,k) Sb(m+1,k+1) Sb(m,k-1) ...
            Sb(m,k+1) Sb(m-1,k-1) Sb(m-1,k) Sb(m-1,k+1)])&Sb(m,k)>=thresh
            Sbp(m,k)=Sb(m,k);
        end
    end
end
for m=1:Nsig
    for k=1:Nw
        Sp(m,k)=Sbp(m+1,k+1);
    end
end
if suppresslastrow==1
    for k=1:Nw
        Sp(Nsig,k)=0;
    end
end
return