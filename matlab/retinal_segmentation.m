function segmentationResults = retinal_segmentation(enhancedImg)
    % RETINAL_SEGMENTATION - Extract Clinically Relevant Retinal Structures
    % Segmentations: Optic Disc, Fovea, Vessels, Microaneurysms (sub-pixel),
    % Exudates, Hemorrhages, Neovascularization.
    %
    % Input:  enhancedImg - RGB enhanced image matrix
    % Output: segmentationResults - Struct containing binary masks & coordinates

    if size(enhancedImg, 3) ~= 3
        error('Input must be 3-channel RGB image.');
    end
    
    imgDouble = im2double(enhancedImg);
    greenChan = imgDouble(:,:,2);
    redChan   = imgDouble(:,:,1);
    
    [rows, cols] = size(greenChan);
    
    % 1. Optic Disc (OD) Localization & Segmentation
    % OD appears bright circular region in Red/Green channel
    brightMap = imgaussfilt(redChan, 5);
    [maxVal, maxIdx] = max(brightMap(:));
    [odCy, odCx] = ind2sub([rows, cols], maxIdx);
    
    % Create Circular OD Mask (~1/10th image width radius)
    [X, Y] = meshgrid(1:cols, 1:rows);
    odRadius = round(min(rows, cols) * 0.08);
    odMask = ((X - odCx).^2 + (Y - odCy).^2) <= odRadius^2;
    
    % 2. Fovea Center Localization
    % Fovea is dark region ~2.5 disc diameters temporal to Optic Disc
    % Search region relative to OD center
    searchDist = round(odRadius * 3.5);
    foveaSearchMask = zeros(rows, cols, 'logical');
    minY = max(1, odCy - odRadius*2);
    maxY = min(rows, odCy + odRadius*2);
    minX = max(1, odCx - searchDist - odRadius*2);
    maxX = min(cols, odCx + searchDist + odRadius*2);
    foveaSearchMask(minY:maxY, minX:maxX) = true;
    foveaSearchMask(odMask) = false; % Exclude OD
    
    darkMap = imgaussfilt(greenChan, 8);
    darkMap(~foveaSearchMask) = Inf;
    [minVal, minIdx] = min(darkMap(:));
    [fovCy, fovCx] = ind2sub([rows, cols], minIdx);
    
    foveaRadius = round(odRadius * 0.6);
    foveaMask = ((X - fovCx).^2 + (Y - fovCy).^2) <= foveaRadius^2;

    % 3. Vessel Network Extraction (Matched Filtering / Morphological Operator)
    invertedGreen = 1 - greenChan;
    vesselEnhanced = top-hat_vessels(invertedGreen);
    vesselMask = vesselEnhanced > 0.18;
    vesselMask(odMask) = false; % Exclude OD area
    
    % 4. Microaneurysm (MA) Sub-Pixel Detection
    % MAs are small circular dark spots (10-50 um)
    topHatMA = imtophat(invertedGreen, strel('disk', 4));
    topHatMA(vesselMask) = 0; % Suppress vessel tree
    topHatMA(odMask) = 0;
    maMask = topHatMA > 0.22;
    maProps = regionprops(maMask, topHatMA, 'WeightedCentroid', 'Area', 'PixelIdxList');
    
    % Filter MA candidates by size and circularity
    maCount = length(maProps);
    maCoords = zeros(maCount, 2);
    for k = 1:maCount
        maCoords(k, :) = maProps(k).WeightedCentroid;
    end

    % 5. Exudate Segmentation (Hard & Soft Exudates)
    % Exudates are bright yellow lipid deposits in L*a*b* / RGB space
    labImg = rgb2lab(imgDouble);
    luminance = labImg(:,:,1);
    bChannel  = labImg(:,:,3); % Yellow axis
    
    exudateCandidates = (luminance > 65) & (bChannel > 12);
    exudateCandidates(odMask) = false; % OD is bright, must exclude
    exudatesMask = imopen(exudateCandidates, strel('disk', 2));
    exudateProps = regionprops(exudatesMask, 'Area');
    exudateArea = sum([exudateProps.Area]);

    % 6. Hemorrhage Classification
    % Hemorrhages are dark red blotches/flames larger than MAs
    darkBlots = imtophat(invertedGreen, strel('disk', 12));
    darkBlots(vesselMask) = 0;
    darkBlots(odMask) = 0;
    hemorrhageMask = (darkBlots > 0.15) & ~maMask;
    hemorrhageMask = bwareaopen(hemorrhageMask, 15);
    hemorrhageProps = regionprops(hemorrhageMask, 'Area', 'Eccentricity');
    hemorrhageCount = length(hemorrhageProps);

    % 7. Neovascularization Detection
    % Abnormal, thin chaotic vessel proliferation near OD (NVD) or elsewhere (NVE)
    thinVesselFilter = strel('line', 5, 45);
    abnormalVessels = imopen(vesselMask, thinVesselFilter) & ~vesselMask;
    nvMask = imdilate(abnormalVessels, strel('disk', 1));
    isNeovascularizationPresent = sum(nvMask(:)) > 50;

    % Build Results Structure
    segmentationResults = struct();
    segmentationResults.opticDiscCenter = [odCx, odCy];
    segmentationResults.opticDiscRadius = odRadius;
    segmentationResults.opticDiscMask   = odMask;
    segmentationResults.foveaCenter     = [fovCx, fovCy];
    segmentationResults.foveaMask       = foveaMask;
    segmentationResults.vesselMask      = vesselMask;
    segmentationResults.maMask          = maMask;
    segmentationResults.maCount         = maCount;
    segmentationResults.maCoords        = maCoords;
    segmentationResults.exudatesMask    = exudatesMask;
    segmentationResults.exudateArea     = exudateArea;
    segmentationResults.hemorrhageMask  = hemorrhageMask;
    segmentationResults.hemorrhageCount = hemorrhageCount;
    segmentationResults.neovascularizationMask = nvMask;
    segmentationResults.isNeovascularization    = isNeovascularizationPresent;
    
    fprintf('=== RETINAL STRUCTURE SEGMENTATION ===\n');
    fprintf('Optic Disc Center : [%d, %d]\n', odCx, odCy);
    fprintf('Fovea Center      : [%d, %d]\n', fovCx, fovCy);
    fprintf('Microaneurysms    : %d detected\n', maCount);
    fprintf('Exudate Area      : %.1f px^2\n', exudateArea);
    fprintf('Hemorrhages       : %d detected\n', hemorrhageCount);
    fprintf('Neovascularization: %s\n', string(isNeovascularizationPresent));
end

function vesselRes = top-hat_vessels(img)
    % Helper: Directional vessel enhancement
    vesselRes = zeros(size(img));
    angles = 0:15:165;
    for deg = angles
        se = strel('line', 9, deg);
        vesselRes = max(vesselRes, imtophat(img, se));
    end
end
