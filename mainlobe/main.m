clc
clear
close all

load mat-single-100dB.mat
% x=[0,0];
% y=[0,0.7/64];
x=[0];
y=[0];
tgt_num=1;
fftm=fft;
% fftlog=10*log10(fftm+1e-13);
% caponlog=10*log10(capon+1e-13);
% musiclog=10*log10(music+1e-13);
% deeplog=10*log10((deep)+1e-13);
dx=1/512;
dy=1/64;
x_label=-0.5:dx:0.5-dx;
y_label=-0.5:dy:0.5-dy;
% figure;
% mesh(x_label,y_label,fftlog);
% axis([-0.5,0.5,-0.5,0.5,-40,0])
% figure;
% mesh(x_label,y_label,caponlog.');
% axis([-0.5,0.5,-0.5,0.5,-40,0])
% figure;
% mesh(x_label,y_label,musiclog.');
% axis([-0.5,0.5,-0.5,0.5,-40,0])
% figure;
% mesh(x_label,y_label,deeplog);
% axis([-0.5,0.5,-0.5,0.5,-40,0])

fft_peak=peak(fftm,0,1);
capon_peak=peak(capon,0,1).';
music_peak=peak(music,0,1).';
deep_peak=peak(deep,0,1);
a2=fft_peak(:);
[BA,I]=sort(a2);
sz=[64,512];
[row1,col1]=ind2sub(sz,I(end-tgt_num+1:end));
f_est_fft=[y_label(row1);x_label(col1)];

a2=capon_peak(:);
[BA,I]=sort(a2);
sz=[64,512];
[row2,col2]=ind2sub(sz,I(end-tgt_num+1:end));
f_est_capon=[y_label(row2);x_label(col2)];

a2=music_peak(:);
[BA,I]=sort(a2);
sz=[64,512];
[row3,col3]=ind2sub(sz,I(end-tgt_num+1:end));
f_est_music=[y_label(row3);x_label(col3)];

a2=deep_peak(:);
[BA,I]=sort(a2);
sz=[64,512];
[row4,col4]=ind2sub(sz,I(end-tgt_num+1:end));
f_est_deep=[y_label(row4);x_label(col4)];
h=figure()
set(h,'position',[100 100 1000 600]);

for i=1:tgt_num
    h1=stem(y(i),-140,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),10,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    plot(x_label,20*log10((fftm(row1(i),:))+1e-13),'m:.','linewidth',3);
    hold on;
    plot(x_label,10*log10((capon(:,row2(i))+1e-13)),':.','color','#4dbeee','linewidth',3);
    hold on;
    plot(x_label,10*log10((music(:,row3(i))+1e-13)),':.','color','#edb120','linewidth',3);
    hold on;
    plot(x_label,20*log10((deep(row4(i),:)+1e-13)),'k:.','linewidth',3);
end

axis([-0.05 0.1 -140 10])
legend('Periodogram','Capon','MUSIC','2-D ResFreq')
xlabel('f_2/Hz')
ylabel('Normalized PSD/dB')
set(gca,'FontSize',20); % 设置文字大小，同时影响坐标轴标注、图例、标题等。
set(get(gca,'XLabel'),'FontSize',20);%图上文字为8 point或小5号
set(get(gca,'YLabel'),'FontSize',20);
grid on
title('Single Freq., SNR = 100 dB')
alldatacursors = findall(gcf,'type','hggroup');
set(alldatacursors,'FontSize',16)
hold on
h1=plot(-0.005859,-64.42,'rs','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',15);
set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
hold on
h2=plot(0.08008,-58.33,'rs','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',15);
set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
hold on
% h2=plot(-0.0332,-63.33,'rs','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',15);
% set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
x=1


