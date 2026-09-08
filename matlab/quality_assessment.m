function [enhancedImg, qualityReport] = quality_assessment(rawImage)
    % QUALITY_ASSESSMENT - Image Quality Assessment & Adaptive Enhancement
    % Evaluates fundus image adequacy (focus, illumination, field of view).
    % Applies adaptive CLAHE, illumination normalization, and denoising.
    % Generates recapture feedback if image is ungradeable.
    %
    % Inputs:
    %   rawImage - Input RGB fundus image matrix (uint8 or double)
    % Outputs:
    %   enhancedImg   - Processed & enhanced RGB image
    %   qualityReport - Structure containing metrics and gradeability status

    if ischar(rawImage) || isstring(rawImage)
        rawImage = imread(rawImage);
    end
    
    if size(rawImage, 3) ~= 3
        error('Input image must be a 3-channel RGB image.');
    end
    
    rawDouble = im2double(rawImage);
    greenChan = rawDouble(:,:,2);
    
    % 1. Field of View (FOV) Masking
    grayImg = rgb2gray(rawDouble);
    fovMask = grayImg > 0.05;
    fovAreaPct = (sum(fovMask(:)) / numel(fovMask)) * 100;
    
    % 2. Focus Assessment (Tenengrad Gradient Metric)
    [Gx, Gy] = imgradientxy(greenChan, 'Sobel');
    gradMag = sqrt(Gx.^2 + Gy.^2);
    sharpnessScore = mean(gradMag(fovMask));
    
    % 3. Illumination Uniformity & Contrast Assessment
    meanIllum = mean(greenChan(fovMask));
    stdIllum  = std(greenChan(fovMask));
    snrEstimate = meanIllum / (stdIllum + 1e-6);
    
    % 4. Determine Gradeability & Feedback
    isFocusOK = sharpnessScore > 0.035;
    isIllumOK = (meanIllum >= 0.15) && (meanIllum <= 0.85);
    isFovOK   = fovAreaPct >= 40.0;
    
    overallGradeable = isFocusOK && isIllumOK && isFovOK;
    
    feedbackList = {};
    if ~isFocusOK
        feedbackList{end+1} = 'BLUR_DETECTED: Refocus portable fundus camera lens or stabilize patient head rest.';
    end
    if meanIllum < 0.15
        feedbackList{end+1} = 'LOW_ILLUMINATION: Increase LED flash intensity or dilate pupil (tropicamide 0.5%).';
    elseif meanIllum > 0.85
        feedbackList{end+1} = 'OVER_EXPOSURE: Reduce flash intensity or adjust gain.';
    end
    if ~isFovOK
        feedbackList{end+1} = 'POOR_FOV: Re-align camera to center optic disc and foveal macula within 45-degree field.';
    end
    if overallGradeable && isempty(feedbackList)
        feedbackList{end+1} = 'GRADEABLE: Image quality acceptable for automated clinical screening.';
    end

    % 5. Adaptive Image Enhancement Pipeline
    % Step A: Illumination Normalization (Large Gaussian Background Estimate)
    bgEstimate = imgaussfilt(greenChan, 30);
    meanBg = mean(bgEstimate(fovMask));
    normGreen = greenChan - bgEstimate + meanBg;
    normGreen = max(0, min(1, normGreen));
    
    % Step B: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    claheGreen = adapthisteq(im2uint8(normGreen), 'ClipLimit', 0.02, 'TileArraySize', [8 8]);
    claheGreenDouble = im2double(claheGreen);
    
    % Step C: Denoising (Bilateral / Edge-preserving Filter)
    if exist('imbilatfilt', 'file')
        denoisedGreen = imbilatfilt(claheGreenDouble, 0.05, 2);
    else
        denoisedGreen = imgaussfilt(claheGreenDouble, 0.8);
    end
    
    % Reconstruct Enhanced RGB
    enhancedImg = rawDouble;
    enhancedImg(:,:,2) = denoisedGreen;
    % Slight adaptive boost to Red channel for vascular contrast
    enhancedImg(:,:,1) = min(1, enhancedImg(:,:,1) * 1.05);
    
    % Compile Quality Report
    qualityReport = struct();
    qualityReport.isGradeable     = overallGradeable;
    qualityReport.sharpnessScore  = sharpnessScore;
    qualityReport.meanIllum       = meanIllum;
    qualityReport.stdIllum        = stdIllum;
    qualityReport.snrEstimate     = snrEstimate;
    qualityReport.fovAreaPct      = fovAreaPct;
    qualityReport.recaptureAction = feedbackList;
    qualityReport.timestamp       = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    fprintf('=== IMAGE QUALITY REPORT ===\n');
    fprintf('Gradeable Status : %s\n', string(overallGradeable));
    fprintf('Sharpness Score  : %.4f (Threshold: >0.035)\n', sharpnessScore);
    fprintf('Mean Illumination: %.4f (Range: 0.15 - 0.85)\n', meanIllum);
    fprintf('FOV Area         : %.2f%%\n', fovAreaPct);
    fprintf('Feedback         : %s\n', strjoin(feedbackList, ' | '));
end
