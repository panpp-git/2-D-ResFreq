clc
close all
clear

% w1=double(w1(1:1024/8,:));
% w2=double(w2(1:128/8,:));
load filters-skip.mat
M=16;
N=128;
fmat=zeros(64,512,M,N);
plr_m=zeros(M,N);
for j=2:size(w1,1)
    for i=2:size(w2,1)
        ret=[i,j]
        cw=w2(i,:).'*w1(j,:);
        tmp=abs(ifftshift(ifftshift((ifft2(cw,64,512)),2),1));
        temp=max(max(tmp));
        plr=temp;
        [~,index]=max(tmp(:));
        [row,col]=ind2sub(size(tmp),index);
        
        if plr_m(floor((row-1)/4)+1,floor((col-1)/4)+1)<plr
            fmat(:,:,floor((row-1)/4)+1,floor((col-1)/4)+1)=tmp;
            plr_m(floor((row-1)/4)+1,floor((col-1)/4)+1)=plr;
        end
    end
end
save fmat-skip.mat fmat plr_m


% fmat=zeros(64,512,M,N);
% plr_m=zeros(M,N);
% for j=2:size(w1,1)
%     for i=2:size(w2,1)
%         ret=[i,j]
%         cw=w2(i,:).'*w1(j,:);
%         tmp=abs(ifftshift(ifftshift((ifft2(cw,64,512)),2),1));
% %         figure(4)
% %         mesh(tmp);
% %         pause(0.5)
%         temp=peak(tmp,0,0);
%         temp=sort(temp(:));
%         plr=temp(end)/temp(end-1);
%         [~,index]=max(tmp(:));
%         [row,col]=ind2sub(size(tmp),index);
%         
%         if plr_m(floor((row-1)/4)+1,floor((col-1)/4)+1)<plr
%             fmat(:,:,floor((row-1)/4)+1,floor((col-1)/4)+1)=tmp;
%             plr_m(floor((row-1)/4)+1,floor((col-1)/4)+1)=plr;
%         end
%     end
% end
% save fmat-skip-0.mat fmat plr_m
load fmat-skip.mat 
[x,y]=meshgrid(-0.5:1/512:0.5-1/512,-0.5:1/64:0.5-1/64);
axis([-0.5,0.5,-0.5,0.5 0 10]);hold on;box on;
for m=1:2:M
    for n=1:8:N
%         cnt=cnt+1;
        if max(max(abs((fmat(:,:,m,n)))))>0
            point=[m,n]
            bb=(fmat(:,:,m,n))*M*N;
            figure(1);
            mesh(x,y,bb)
            hold on
        end
    end
end
view(45,70)
xlabel('f_2/Hz')
ylabel('f_1/Hz')
set(gca,'FontSize',34); % 设置文字大小，同时影响坐标轴标注、图例、标题等。
set(get(gca,'XLabel'),'FontSize',34);%图上文字为8 point或小5号
set(get(gca,'YLabel'),'FontSize',34);
grid on
xxx=1

% cnt=-1;
% for m=1:2:M
%     figure(m);
%     cnt=cnt+1;
%     for n=1:8:N
%         if max(max(abs((fmat(:,:,m,n)))))>0
%             point=[m,n]
%             bb=(fmat(:,:,m,n))*M*N;
%             plot(-0.5:1/512:0.5-1/512,fmat(5+cnt*8,:,m,n)*M*N,'k','linewidth',1.5)
%             
%             hold on  
%             pause(0.5)
%         end
%     end
%     axis([-0.5 0.5 0 10]);
%     set(gca,'position',[0 0 1 1]);
%     axis off 
%     set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w')
% %     set(gcf,'position',[100,100, 300, 150]); %设定figure的位置和大小 get current figure
%     colormap jet
%     path=strcat('M',num2str(m),'.bmp');
%     saveas(gcf,path);
% end
% close all

figure;
xx=-0.5:1/16:0.5-1/16;
for m=1:2:16
    path=strcat('M',num2str(m),'.bmp');
    im=imread(path);
    gray_im=im(:,:,1);
    gray_im = gray_im(end:-1:1, 1:1:end);
    [yy,zz]=meshgrid(-0.5:1/size(gray_im,2):0.5-1/size(gray_im,2),0:10/size(gray_im,1):10-10/size(gray_im,1));
    z=double(gray_im)/255;
    b=surf(yy,z-z+xx(m),zz,double(gray_im),'FaceAlpha','flat');
    set(b,'alphadata',1-z)
    shading interp;
    colormap(gray);
    set(b,'linestyle','none');
    hold on
end
axis([-0.5 0.5 -0.5-1/16 0.5 0 10]);

view(45,75);
xlabel('f_2/Hz')
ylabel('f_1/Hz')
set(gca,'FontSize',30); % 设置文字大小，同时影响坐标轴标注、图例、标题等。
set(get(gca,'XLabel'),'FontSize',30);%图上文字为8 point或小5号
set(get(gca,'YLabel'),'FontSize',30);
grid on

% cnt=-1;
% for n=1:8:N
%     cnt=cnt+1;
%     figure(n);
%     for m=1:2:M
%         if max(max(abs((fmat(:,:,m,n)))))>0
%             point=[m,n]
%             bb=(fmat(:,:,m,n))*M*N;
%             plot(-0.5:1/64:0.5-1/64,fmat(:,32*cnt+3,m,n)*M*N,'k','linewidth',1.5)
%             
%             hold on  
%             pause(0.5)
%         end
%     end
%     axis([-0.5 0.5 0 10]);
%     set(gca,'position',[0 0 1 1]);
%     axis off 
%     set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w')
%     colormap jet
%     path=strcat('N',num2str(n),'.bmp');
%     saveas(gcf,path);
% end
% close all

figure;
yy=-0.5:1/N:0.5-1/N;
for n=1:16:N
    path=strcat('N',num2str(n),'.bmp');
    im=imread(path);
    gray_im=im(:,:,1);
    gray_im = gray_im(end:-1:1, 1:1:end);
    [xx,zz]=meshgrid(-0.5:1/size(gray_im,2):0.5-1/size(gray_im,2),0:10/size(gray_im,1):10-10/size(gray_im,1));
    z=double(gray_im)/255;
    b=surf(z-z+yy(n),xx,zz,double(gray_im),'FaceAlpha','flat');
    set(b,'alphadata',1-z)
    shading interp;
    colormap(gray);
    set(b,'linestyle','none');
    hold on
end
axis([-0.5-1/16 0.5 -0.5 0.5 0 10]);

view(45,75);
xlabel('f_2/Hz')
ylabel('f_1/Hz')
set(gca,'FontSize',34); % 设置文字大小，同时影响坐标轴标注、图例、标题等。
set(get(gca,'XLabel'),'FontSize',34);%图上文字为8 point或小5号
set(get(gca,'YLabel'),'FontSize',34);
grid on





