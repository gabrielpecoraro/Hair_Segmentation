Image = imread('mehdi.png');
mask = imread('result.png');

MaskRGB = cat(3, mask, mask, mask);

rgbImage = double(Image);
rgbmask = double(mask);
hsvMask = rgb2hsv(MaskRGB);
hsvImage = rgb2hsv(rgbImage);
hImage = hsvImage(:, :, 1);
sImage = hsvImage(:, :, 2);
vImage = hsvImage(:, :, 3);

hMask = hsvMask(:, :, 1);
sMask = hsvMask(:, :, 2);
vMask = hsvMask(:, :, 3);


moy = 0;
m = 0;
for i=1:256
    for j=1:256
        if hsvMask(i,j,3) > 0
            moy = moy + mod(hsvImage(i,j,3), 256);
            hsvImage(i,j,3);
            m = m+1;
        end
    end
end

moy = moy/m;

hsvImage();

figure,
subplot(2,2,1),
hHist = histogram(hImage);
grid on;
title('Hue Histogram');
subplot(2,2,2);
sHist = histogram(sImage);
grid on;
title('Saturation Histogram');
subplot(2,2,3);
vHist = histogram(vImage);
grid on;
title('Value Histogram');
subplot(2,2,4);
imshow(Image);

h1Image = hsvImage(:, :, 1);
s1Image = hsvImage(:, :, 2);
v1Image = hsvImage(:, :, 3);

h2Image = hsvImage(:, :, 1);
s2Image = hsvImage(:, :, 2);
v2Image = hsvImage(:, :, 3);

h3Image = hsvImage(:, :, 1);
s3Image = hsvImage(:, :, 2);
v3Image = hsvImage(:, :, 3);

h4Image = hsvImage(:, :, 1);
s4Image = hsvImage(:, :, 2);
v4Image = hsvImage(:, :, 3);

for i=1:256
    for j=1:256
        if hsvMask(i,j,3) > 0
            h1Image(i,j) = 0;
            v1Image(i,j) = moy - mod(v1Image(i,j), 256);
            h2Image(i,j) = 0;
            v2Image(i,j) = 255;
            h2Image(i,j) = 0;
            v3Image(i,j) =+ 255;
            h3Image(i,j) = 0;
            v4Image(i,j) =+ 120;
            h4Image(i,j) = 0;
        end
    end
end


figure,
subplot(2,2,1),
histogram(v1Image);
grid on;
title('Hue Histogram');
subplot(2,2,2);
histogram(v2Image);
grid on;
title('Hue Histogram');
subplot(2,2,3);
histogram(v3Image);
grid on;
title('Hue Histogram');
subplot(2,2,4);
histogram(v4Image);
title('Hue Histogram');


HSV1Image(:,:,1) = h1Image(:,:);
HSV1Image(:,:,2) = s1Image(:,:);
HSV1Image(:,:,3) = v1Image(:,:);

HSV2Image(:,:,1) = h2Image(:,:);
HSV2Image(:,:,2) = s2Image(:,:);
HSV2Image(:,:,3) = v2Image(:,:);

HSV3Image(:,:,1) = h3Image(:,:);
HSV3Image(:,:,2) = s3Image(:,:);
HSV3Image(:,:,3) = v3Image(:,:);

HSV4Image(:,:,1) = h4Image(:,:);
HSV4Image(:,:,2) = s4Image(:,:);
HSV4Image(:,:,3) = v4Image(:,:);

RGB1Image = hsv2rgb(HSV1Image);
RGB2Image = hsv2rgb(HSV2Image);
RGB3Image = hsv2rgb(HSV3Image);
RGB4Image = hsv2rgb(HSV4Image);


figure,
subplot(2,2,1),
imshow(HSV1Image);
grid on;
title('Hue Histogram');
subplot(2,2,2);
imshow(HSV2Image);
grid on;
title('Saturation Histogram');
subplot(2,2,3);
imshow(HSV3Image);
grid on;
title('Value Histogram');
subplot(2,2,4);
imshow(HSV4Image);

