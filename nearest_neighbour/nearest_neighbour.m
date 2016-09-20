clear all; clf;
figure(1);
colormap(gray(256));



best_dist = inf;
worst_dist = -1;
location = 'C:\Users\roans\Desktop\thesis\MATLAB\tiffset';

filenames = [];
file_list = 'filenames.txt';
file_list = fullfile(location,file_list);
fid = fopen(file_list,'r');

tline = fgetl(fid);
while ischar(tline(1))
   filenames = [filenames;[tline,'.tif']]; 
   tline = fgetl(fid);
end
fclose(fid);

infile = datasample(filenames,1);
infile_full = fullfile(location,infile);
in_img = Tiff(infile_full, 'r');

inx = in_img.getTag('ImageWidth');
iny = in_img.getTag('ImageLength');
in_img_arr = in_img.read();
in_img_arr_temp = in_img_arr;
subplot(2,2,1);
imagesc(in_img_arr);
title('Input Image')
axis image
axis off

correct_count = 0;
total = 0;
for k = 1:length(filenames)
    best_dist = inf;
    worst_dist = -1;
    infile = filenames(k,:);
    infile_full = fullfile(location,infile);
    in_img = Tiff(infile_full, 'r');

    inx = in_img.getTag('ImageWidth');
    iny = in_img.getTag('ImageLength');
    in_img_arr = in_img.read();
    in_img_arr_temp = in_img_arr;
    
    subplot(2,2,1);
    imagesc(in_img_arr_temp);
    title('Input Image')
    axis image
    axis off
    for n = 1:length(filenames)
        in_img_arr_temp = in_img_arr;
        col=inx;  row=iny;
        fname = filenames(n,:);
%         display([infile,' ', num2str(inx),'x', num2str(iny)]);
    %     filename = strcat('HB0',num2str(imagenum),'.003.tiff');
        fname_full = fullfile(location, filenames(n,:));
    %     if(exist(fname,'file') == 2)
        if (~strcmp(fname,infile))
            img = Tiff(fname_full, 'r');
            img_arr = img.read();

%             display([fname,' ', num2str(img.getTag('ImageWidth')),'x', num2str(img.getTag('ImageLength'))]);

            if(img.getTag('ImageLength') > iny || img.getTag('ImageWidth') > inx)

    %             display(['resizing to ',num2str(inx),' by ',num2str(iny)])
                crop_rect = [abs((inx - col))/2 abs((iny - row))/2 col row];
                img_arr = imcrop(img_arr,crop_rect);

            else
                row = img.getTag('ImageLength');
                col = img.getTag('ImageWidth');
    %             display(['resizing2 to ',num2str(col),' by ',num2str(row)])
                crop_rect = [abs((inx - col))/2 abs((iny - row))/2 col row];
                in_img_arr_temp = imcrop(in_img_arr,crop_rect);
            end

            
            
            dist = 0;
            for y = 1:row        
                for x = 1:col
                    pixel_dist = (int32(in_img_arr_temp(y,x)) - int32(img_arr(y,x)))^2;
                    dist = dist + pixel_dist;
                end
            end
            dist = dist/(row*col);
            
            if (dist <= best_dist)
                best_dist = dist;
                closest_img = fname;
            end

            if (dist >= worst_dist)
                worst_dist = dist;
                furthest_img = fname;
            end
            
            close(img);
            
        end
    end
    
    if(strcmp(closest_img(9:11),infile(9:11)))
        disp('Correct!')
        disp(['Input file: ',infile]);
        disp(['Best guess: ',closest_img]);
        disp(['Worst guess: ',furthest_img]);
        disp([closest_img(9:11),' = ',infile(9:11)])
        correct_count = correct_count + 1;
    else
        disp('Incorrect!')
    end
    total = total + 1;
    disp([num2str(correct_count),'/',num2str(total),' correct']);
    close(in_img);
    
    
    best_guess = closest_img;
    best_guess = fullfile(location,best_guess);
    best_guess = Tiff(best_guess, 'r');
    best_guess_arr = best_guess.read();
    close(best_guess);
    worst_guess = furthest_img;
    worst_guess = fullfile(location,worst_guess);
    worst_guess = Tiff(worst_guess, 'r');
    worst_guess_arr = worst_guess.read();
    close(worst_guess)

    subplot(2,2,2);
    imagesc(best_guess_arr);
    title('Best Guess')
    axis image
    axis off

    subplot(2,2,3);
    imagesc(worst_guess_arr);
    title('Worst Guess')
    axis image
    axis off
end
%%% display best match
% best_guess = closest_img;
% best_guess = fullfile(location,best_guess);
% best_guess = Tiff(best_guess, 'r');
% best_guess_arr = best_guess.read();
% close(best_guess);
% worst_guess = furthest_img;
% worst_guess = fullfile(location,worst_guess);
% worst_guess = Tiff(worst_guess, 'r');
% worst_guess_arr = worst_guess.read();
% cloe(worst_guess)
% 
% subplot(2,2,2);
% imagesc(best_guess_arr);
% title('Best Guess')
% axis image
% axis off
% 
% subplot(2,2,3);
% imagesc(worst_guess_arr);
% title('Worst Guess')
% axis image
% axis off
% 
% disp(['Input file: ',infile]);
% disp(['Best guess: ',closest_img]);
% disp(['Worst guess: ',furthest_img]);
% disp([closest_img(9:11),' = ',infile(9:11)])
% if(strcmp(closest_img(9:11),infile(9:11)))
%     disp('Correct!')
% else
%     disp('Incorrect!')
% end