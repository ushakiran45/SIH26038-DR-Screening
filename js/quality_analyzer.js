/**
 * SIH26038 - Image Quality Assessment & Adaptive Enhancement Module
 * Computes Tenengrad sharpness, illumination uniformity, SNR, FOV area,
 * and generates actionable recapture feedback for rural PHC operators.
 */

window.QualityAnalyzer = (function() {

    function analyzeQuality(sampleKey) {
        const sample = window.FundusEngine.SAMPLES[sampleKey] || window.FundusEngine.SAMPLES['moderate_npdr'];

        const sharpness = sample.sharpness;
        const meanIllum = sample.meanIllum;
        const fovPct = sample.fovPct;

        const isFocusOK = sharpness > 0.035;
        const isIllumOK = meanIllum >= 0.15 && meanIllum <= 0.85;
        const isFovOK = fovPct >= 40.0;

        const isGradeable = isFocusOK && isIllumOK && isFovOK;

        const feedbackList = [];
        if (!isFocusOK) {
            feedbackList.push({
                type: 'warning',
                code: 'BLUR_DETECTED',
                title: 'Out of Focus / Motion Blur Detected',
                text: 'Re-stabilize patient chin rest. Clean portable fundus camera lens with optical tissue and adjust focus wheel.'
            });
        }
        if (meanIllum < 0.15) {
            feedbackList.push({
                type: 'danger',
                code: 'LOW_ILLUMINATION',
                title: 'Severe Under-Exposure',
                text: 'Increase LED flash level on camera. Administer tropicamide 0.5% drops to dilate pupil if undilated.'
            });
        } else if (meanIllum > 0.85) {
            feedbackList.push({
                type: 'warning',
                code: 'OVER_EXPOSURE',
                title: 'Over-Exposure / Flash Glare',
                text: 'Reduce LED flash intensity slider by 20% to prevent macular blooming.'
            });
        }
        if (!isFovOK) {
            feedbackList.push({
                type: 'warning',
                code: 'POOR_FOV',
                title: 'Inadequate Field of View (<40%)',
                text: 'Re-position camera lens closer to cornea. Ensure 45-degree posterior pole coverage.'
            });
        }

        if (isGradeable && feedbackList.length === 0) {
            feedbackList.push({
                type: 'success',
                code: 'GRADEABLE_PASS',
                title: 'Adequate Quality Confirmed',
                text: 'Image exceeds clarity & illumination thresholds for automated DR screening.'
            });
        }

        return {
            sampleName: sample.name,
            isGradeable,
            sharpness: (sharpness * 100).toFixed(2),
            meanIllum: (meanIllum * 100).toFixed(1),
            fovPct: fovPct.toFixed(1),
            snrEstimate: (meanIllum / 0.04).toFixed(1) + ' dB',
            feedbackList
        };
    }

    function updateQualityUI(sampleKey) {
        const result = analyzeQuality(sampleKey);

        const statusBadge = document.getElementById('iqa-status-badge');
        const sharpnessVal = document.getElementById('iqa-sharpness');
        const illumVal = document.getElementById('iqa-illum');
        const fovVal = document.getElementById('iqa-fov');
        const snrVal = document.getElementById('iqa-snr');
        const feedbackContainer = document.getElementById('iqa-feedback-container');

        if (statusBadge) {
            if (result.isGradeable) {
                statusBadge.className = 'badge badge-live';
                statusBadge.innerHTML = '<span class="status-dot"></span> GRADEABLE (PASS)';
            } else {
                statusBadge.className = 'badge';
                statusBadge.style.background = 'rgba(255, 82, 82, 0.2)';
                statusBadge.style.color = '#ff5252';
                statusBadge.style.borderColor = '#ff5252';
                statusBadge.innerHTML = '⚠️ UNGRADEABLE (RECAPTURE NEEDED)';
            }
        }

        if (sharpnessVal) sharpnessVal.innerText = result.sharpness;
        if (illumVal) illumVal.innerText = result.meanIllum + '%';
        if (fovVal) fovVal.innerText = result.fovPct + '%';
        if (snrVal) snrVal.innerText = result.snrEstimate;

        if (feedbackContainer) {
            feedbackContainer.innerHTML = result.feedbackList.map(item => `
                <div class="feedback-box ${item.type === 'success' ? '' : (item.type === 'danger' ? 'danger' : 'warning')}">
                    <div class="feedback-title" style="color: ${item.type === 'success' ? '#00e676' : (item.type === 'danger' ? '#ff5252' : '#ffb300')}">
                        ${item.type === 'success' ? '✓' : '⚠️'} ${item.title}
                    </div>
                    <div class="feedback-text">${item.text}</div>
                </div>
            `).join('');
        }
    }

    return {
        analyzeQuality,
        updateQualityUI
    };
})();
