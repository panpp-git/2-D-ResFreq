function [tt]=ref_func_res(tgt_num)
    tgt_num = double(h5read('tgt_num_acc.h5','/tgt_num_acc'));
    sig = (h5read('signal_acc.h5','/signal_acc'));
    sig_real=sig(:,:,1).';
    sig_imag=sig(:,:,2).';
    sig=sig_real+1j*sig_imag;
    %y=abs(fftshift(ifft(ifft(sig,512,2),64,1)));
    %y_max=max(max(abs(y)));
    %figure;mesh(((y/y_max)));
    RPP=sig;
    siz=size(RPP);
    M=siz(1,1);
    N=siz(1,2);
    n1=ceil(2*N/5);m1=ceil(2*M/5);
    dim=n1*m1;
    fblooknum=(N-n1+1)*(M-m1+1);

    dx=1/(8*N);
    dy=1/(8*M);
    x_label=-0.5:dx:0.5-dx;
    y_label=-0.5:dy:0.5-dy;
    t=clock;
    x=0;
    X1=zeros(dim,fblooknum);
    for b=1:(N-n1+1)
        for a=1:(M-m1+1)
          B1=RPP((a:(a+m1-1)),(b:(b+n1-1)));
          x=x+1;
          z1=B1(:);
          X1(:,x)=z1;
        end
    end

    Rxx = X1*X1'/tgt_num;
    [EV,D] = eig(Rxx);
    [EVA,I] = sort(diag(D).');
    EV = fliplr(EV(:,I));
    Un = EV(:,tgt_num+1:end);

    NX=Un';
    bb=zeros(length(x_label),length(y_label));
    for L=1:size(NX,1)
        temp=reshape(NX(L,:),m1,n1).';
        aa=ifft2(temp,length(x_label),length(y_label));
        bb=bb+conj(aa).*aa;
    end
    SP1=1./fftshift(bb);
    music_t=etime(clock,t);
    %figure;mesh(SP1)
    music_spec=SP1;


    load steering_for_capon.mat
    t=clock;
    inv_R=inv(Rxx);
    cc=zeros(length(x_label),length(y_label));
    for L=1:size(inv_R,1)
        temp=reshape(inv_R(L,:),m1,n1).';
        rret=fftshift(ifft2(temp,length(x_label),length(y_label)));
        cc=cc+conj(aa(:,:,L)).*rret;
    end
    CAPON=abs(1./(cc));
    capon_t=etime(clock,t);
    tt=[music_t,capon_t];
    %figure;mesh(CAPON)


    SP=SP1.';
    Sp2=peak(SP,0,0);

    a2=Sp2(:);
    [BA,I]=sort(a2);
    sz=[8*M,8*N];
    [row,col]=ind2sub(sz,I(end-tgt_num+1:end));
    f_est_music=[y_label(row);x_label(col)];

    SP=CAPON.';
    Sp2=peak(SP,0,0);

    a2=Sp2(:);
    [BA,I]=sort(a2);
    sz=[8*M,8*N];
    [row,col]=ind2sub(sz,I(end-tgt_num+1:end));
    f_est_capon=[y_label(row);x_label(col)];



    if ~exist('f_est_capon_acc.h5','file')==0
        delete('f_est_capon_acc.h5')
    end

    if ~exist('f_est_music_acc.h5','file')==0
        delete('f_est_music_acc.h5')
    end
    h5create('f_est_music_acc.h5','/f_est_music_acc',size(music_spec));
    h5write('f_est_music_acc.h5','/f_est_music_acc',music_spec);
    h5create('f_est_capon_acc.h5','/f_est_capon_acc',size(CAPON));
    h5write('f_est_capon_acc.h5','/f_est_capon_acc',CAPON);
end