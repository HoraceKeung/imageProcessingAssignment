function result = colourMatrix(filename)
    %% S1 read image
    im = imread(filename);
    imOri = imread('Archive/org_1.png');
    
    %% S2 correction of white balance
    %http://uk.mathworks.com/matlabcentral/fileexchange/41089-color-balance-demo-with-gpu-computing/content/GPUDemoCode/whitebalance.m
    % WHITEBALANCE forces the average image color to be gray
    % Copyright 2013 The MathWorks, Inc.
    
    % Find the average values for each channel
    pageSize = size(im,1) * size(im,2);
    avg_rgb = mean( reshape(im, [pageSize,3]) );
    % Find the average gray value and compute the scaling array
    avg_all = mean(avg_rgb);
    scaleArray = max(avg_all, 128)./avg_rgb;
    scaleArray = reshape(scaleArray,1,1,3);
    % Adjust the image to the new gray value
    im = uint8(bsxfun(@times,double(im),scaleArray));
    
    %% S3 set any pixel with RGB value greater than certain value to white
    im(im>200)=255;

    %% S4 reduce saturation and turn image into binary
    HSV1 = rgb2hsv(im);
    HSV1(:,:,2) = HSV1(:,:,2)/2;
    HSV1(HSV1 < 0) = 0;
    imLowSat = hsv2rgb(HSV1);
    bi = im2bw(imLowSat,0.3);

    %% S5 remove salt and pepper noise
    biNoSalt = filter2(fspecial('average',8),bi);
    biNoSalt = wiener2(biNoSalt,[6 6]);
    biNoSalt = im2bw(biNoSalt,0.6);

    %% S6 line dilate
    se = strel('line',2,0);
    biDi = imdilate(biNoSalt,se);
    se = strel('line',2,90);
    biDi = imdilate(biDi,se);
    
    %% S7 disk dilate
    se = strel('disk',2);
    biDi = imdilate(biDi,se);

    %% S8 disk imerode
    se = strel('disk',4);
    biEr = imerode(biDi,se);
    
    %% S9 complement the binary image
    biCo = imcomplement(biEr);

    %% S10 Extract objects from binary image by size, region size must be between certain range
    biAf = bwareafilt(biCo, [200 13000]);
    
    %% S11 crop image
    stats = regionprops('table',biAf,'BoundingBox');
    box = stats.BoundingBox;
    %add fifth column storing x + width
    box = [box box(:,1)+box(:,3)];
    %add sixth column storing y + height
    box = [box box(:,2)+box(:,4)];
    box = sortrows(box, 1);
    minx = box(1,1);
    box = sortrows(box, 2);
    miny = box(1,2);
    box = sortrows(box, 5);
    maxx = box(end,5);
    box = sortrows(box, 6);
    maxy = box(end,6);
    newWidth = maxx-minx;
    newHeight = maxy-miny;
    biCrop = imcrop(biAf,[minx miny newWidth newHeight]);
    im = imcrop(im,[minx miny newWidth newHeight]);
    
    %% S12 Circle and rectangle detection
    stats = regionprops('table',biCrop,'Centroid','Eccentricity','Area','MajorAxisLength','MinorAxisLength');
    %sort by area and get the ten biggest regions.
    stats = sortrows(stats,'Area');
    stats = stats(max(end-9,1):end,:); 
    %find the six furthest regions away from the average of regions’ centroids
    centre = mean(stats.Centroid);
    temp = [];
    C = stats.Centroid;
    for i = 1:height(stats)
        X = [C(i,1),C(i,2);centre(1),centre(2)];
        temp(end+1) = pdist(X,'euclidean');
    end
    temp = temp(:);
    stats.DistFromCentre = temp;
    stats = sortrows(stats,'DistFromCentre');
    stats = stats(max(end-5,1):end,:); 
    %sort by Eccentricity and get the 2 regions that have the largest value, which have the highest chance of being the rectangles. The rest will be circles.
    stats = sortrows(stats,'Eccentricity');
    statsForRect = stats(end-1:end,:); 
    statsForCir = stats(1:end-2,:); 
    %find the circle that is closest to the two rectangles. This circle should be the one at the top left corner.
    temp2 = [];
    C = statsForCir.Centroid;
    R = statsForRect.Centroid;
    for i = 1:height(statsForCir)
        D = 0;
        for j = 1:height(statsForRect)
            X = [C(i,1),C(i,2);R(j,1),R(j,2)];
            D = D + pdist(X,'euclidean');
        end
        temp2(end+1) = D;
    end
    temp2 = temp2(:);
    statsForCir.DistFromRect = temp2;
    statsForCir = sortrows(statsForCir,'DistFromRect');
    statsForCirClosestToRect = statsForCir(1,:); 
    
    %show image with found circles
    figure(1);
    imshow(biCrop);
    diameters = mean([statsForCir.MajorAxisLength statsForCir.MinorAxisLength],2);
    radii = diameters/2;
    hold on
    viscircles(statsForCir.Centroid,radii);
    hold off
    
    %% S13 Recover the image
    %for each corner, find the closest circle and map them to each other.
    Corners = [1 1;1 newWidth; newHeight newWidth; newHeight 1];
    badPoints = [];
    degreeNeedToRotate = 0;
    for i = 1:4
        closest = C(1,:);
        D = pdist([C(1,:);Corners(i,:)], 'euclidean');
        for j = 2:4
            if pdist([C(j,:);Corners(i,:)], 'euclidean') < D
                closest = C(j,:);
                D = pdist([C(j,:);Corners(i,:)], 'euclidean');
            end
        end
        badPoints = [badPoints;closest];
        if closest == statsForCirClosestToRect.Centroid
            degreeNeedToRotate = (i-1) * -90;
        end
    end
    %correct positions of the four circles
    correctC1 = [24.5 24.5];
    correctC2 = [24.5 444.75];
    correctC3 = [444.75 444.75];
    correctC4 = [444.75 24.5];
    CorrectPoints = [correctC1; correctC2; correctC3; correctC4];
    tform = estimateGeometricTransform(badPoints, CorrectPoints, 'projective');
    outputView = imref2d(size(imOri));
    output  = imwarp(im,tform,'OutputView',outputView);
    %once the circles are at the right place, rotate the image accordingly.
    im = imrotate(output, degreeNeedToRotate);
    
    %% S14 increase colour saturation and blur the image using gaussian filter
    im = imgaussfilt(im,5);
    im = im2double(im);
    HSV = rgb2hsv(im);
    HSV(:,:,2) = HSV(:,:,2)*3;
    HSV(HSV > 1) = 1;
    im = hsv2rgb(HSV);
    im = imgaussfilt(im,5);
    figure(2);
    imshow(im);
    
    %% S15 Generate result
    result = char(4,4);
    %distance from the edge of the image to the first cell
    offset_x = 30.5;
    offset_y = 30.5;
    square_size = 105;
    
    for x = 1:4
       for y = 1:4
          px = offset_x + ((x-0.5)*square_size);
          py = offset_y + ((y-0.5)*square_size);
          R = [im(px,py,1),im(px,py,2),im(px,py,3);1,0,0];
          redDist = pdist(R, 'euclidean');
          G = [im(px,py,1),im(px,py,2),im(px,py,3);0,1,0];
          greenDist = pdist(G, 'euclidean');
          B = [im(px,py,1),im(px,py,2),im(px,py,3);0,0,1];
          blueDist = pdist(B, 'euclidean');
          Y = [im(px,py,1),im(px,py,2),im(px,py,3);1,1,0];
          yellowDist = pdist(Y, 'euclidean');
          W = [im(px,py,1),im(px,py,2),im(px,py,3);1,1,1];
          whiteDist = pdist(W, 'euclidean');
          dists = [redDist,greenDist,blueDist,yellowDist,whiteDist];
          colours = {'R','G','B','Y','W'};
          mapObj = containers.Map(dists,colours);
          closestDist = min(dists);
          result(x,y) = mapObj(closestDist);
       end
    end
end
