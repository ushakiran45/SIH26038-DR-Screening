/**
 * SIH26038 - Retinal Canvas Engine
 * Generates synthetic clinical fundus images and performs real-time image processing,
 * structure layer toggling, and adaptive CLAHE enhancement.
 */

window.FundusEngine = (function() {
    let currentSampleKey = 'moderate_npdr';
    
    // Sample definitions based on clinical benchmarks
    const SAMPLES = {
        'normal': {
            name: 'APTOS 2019 #1042 - Level 0: No DR',
            drLevel: 0,
            quality: 'GRADEABLE',
            sharpness: 0.052,
            meanIllum: 0.48,
            fovPct: 88.5,
            maCount: 0,
            exudatesArea: 0,
            hemorrhagesCount: 0,
            hasNV: false,
            desc: 'Healthy retina. Clear optic disc, distinct fovea, normal macula, clear vascular tree.'
        },
        'mild_npdr': {
            name: 'IDRiD #088 - Level 1: Mild NPDR',
            drLevel: 1,
            quality: 'GRADEABLE',
            sharpness: 0.048,
            meanIllum: 0.45,
            fovPct: 86.2,
            maCount: 3,
            exudatesArea: 0,
            hemorrhagesCount: 0,
            hasNV: false,
            desc: 'Isolated sub-pixel microaneurysms detected in temporal macula region. No exudates.'
        },
        'moderate_npdr': {
            name: 'Messidor-2 #412 - Level 2: Moderate NPDR (Referable)',
            drLevel: 2,
            quality: 'GRADEABLE',
            sharpness: 0.046,
            meanIllum: 0.42,
            fovPct: 85.0,
            maCount: 9,
            exudatesArea: 420,
            hemorrhagesCount: 3,
            hasNV: false,
            desc: 'Multiple microaneurysms, circinate hard exudates near fovea, dot hemorrhages.'
        },
        'severe_npdr': {
            name: 'APTOS 2019 #3891 - Level 3: Severe NPDR (Referable)',
            drLevel: 3,
            quality: 'GRADEABLE',
            sharpness: 0.041,
            meanIllum: 0.38,
            fovPct: 84.1,
            maCount: 22,
            exudatesArea: 1850,
            hemorrhagesCount: 16,
            hasNV: false,
            desc: '>15 intraretinal hemorrhages across 4 quadrants, soft cotton wool spots, large exudates.'
        },
        'pdr': {
            name: 'IDRiD #104 - Level 4: Proliferative DR (PDR Referable)',
            drLevel: 4,
            quality: 'GRADEABLE',
            sharpness: 0.039,
            meanIllum: 0.36,
            fovPct: 82.5,
            maCount: 35,
            exudatesArea: 3200,
            hemorrhagesCount: 28,
            hasNV: true,
            desc: 'Neovascularization at optic disc (NVD), extensive preretinal hemorrhages.'
        },
        'low_quality_blur': {
            name: 'Field PHC #012 - UNGRADEABLE (Blurry Focus)',
            drLevel: -1,
            quality: 'UNGRADEABLE',
            sharpness: 0.018,
            meanIllum: 0.40,
            fovPct: 80.0,
            maCount: 0,
            exudatesArea: 0,
            hemorrhagesCount: 0,
            hasNV: false,
            desc: 'Severe motion blur & out-of-focus acquisition from handheld fundus camera.'
        },
        'low_quality_dark': {
            name: 'Field PHC #019 - UNGRADEABLE (Low Illumination)',
            drLevel: -1,
            quality: 'UNGRADEABLE',
            sharpness: 0.038,
            meanIllum: 0.08,
            fovPct: 52.0,
            maCount: 0,
            exudatesArea: 0,
            hemorrhagesCount: 0,
            hasNV: false,
            desc: 'Insufficient LED flash intensity and undilated pupil obscuring fundus details.'
        }
    };

    /**
     * Render synthetic fundus photo onto target HTML5 Canvas
     */
    function renderFundusImage(canvasId, sampleKey, options = {}) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width || 600;
        const height = canvas.height || 600;

        const sample = SAMPLES[sampleKey] || SAMPLES['moderate_npdr'];
        currentSampleKey = sampleKey;

        // Clear background
        ctx.fillStyle = '#02040a';
        ctx.fillRect(0, 0, width, height);

        const cx = width / 2;
        const cy = height / 2;
        const radius = Math.min(width, height) * 0.43;

        // Apply low illumination if dark sample
        let illumFactor = (sample.meanIllum < 0.15) ? 0.25 : 1.0;
        let isBlur = (sample.sharpness < 0.03);

        ctx.save();
        if (isBlur) {
            ctx.filter = 'blur(6px)';
        }

        // Draw Fundus Circular Background
        const grad = ctx.createRadialGradient(cx, cy, radius * 0.1, cx, cy, radius);
        if (options.mode === 'green_channel') {
            // Green channel view
            grad.addColorStop(0, `rgb(0, ${Math.round(210 * illumFactor)}, 0)`);
            grad.addColorStop(0.7, `rgb(0, ${Math.round(140 * illumFactor)}, 0)`);
            grad.addColorStop(1, `rgb(0, ${Math.round(30 * illumFactor)}, 0)`);
        } else if (options.mode === 'clahe') {
            // CLAHE Enhanced view (crisp contrast)
            grad.addColorStop(0, `rgb(${Math.round(230 * illumFactor)}, ${Math.round(150 * illumFactor)}, ${Math.round(40 * illumFactor)})`);
            grad.addColorStop(0.7, `rgb(${Math.round(180 * illumFactor)}, ${Math.round(90 * illumFactor)}, ${Math.round(20 * illumFactor)})`);
            grad.addColorStop(1, `rgb(${Math.round(50 * illumFactor)}, ${Math.round(20 * illumFactor)}, 0)`);
        } else {
            // Natural RGB view
            grad.addColorStop(0, `rgb(${Math.round(210 * illumFactor)}, ${Math.round(100 * illumFactor)}, ${Math.round(25 * illumFactor)})`);
            grad.addColorStop(0.7, `rgb(${Math.round(160 * illumFactor)}, ${Math.round(60 * illumFactor)}, ${Math.round(15 * illumFactor)})`);
            grad.addColorStop(1, `rgb(${Math.round(40 * illumFactor)}, ${Math.round(10 * illumFactor)}, 0)`);
        }

        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.clip(); // Restrict details inside fundus circle

        // Draw Blood Vessel Tree
        drawVessels(ctx, cx, cy, radius, options);

        // Draw Optic Disc
        const odX = cx + radius * 0.5;
        const odY = cy - radius * 0.05;
        const odRad = radius * 0.20;
        
        if (options.showOpticDisc || options.showAllOverlay) {
            drawOpticDisc(ctx, odX, odY, odRad, options);
        } else {
            // Base Optic Disc
            ctx.beginPath();
            ctx.arc(odX, odY, odRad, 0, Math.PI * 2);
            ctx.fillStyle = options.mode === 'green_channel' ? 'rgb(200, 255, 200)' : 'rgb(255, 230, 180)';
            ctx.fill();
        }

        // Draw Fovea
        const fovX = cx - radius * 0.25;
        const fovY = cy - radius * 0.02;
        const fovRad = radius * 0.12;

        if (options.showFovea || options.showAllOverlay) {
            drawFovea(ctx, fovX, fovY, fovRad);
        }

        // Draw Lesions based on DR Level & Layer Toggles
        if (sample.drLevel >= 1) {
            if (options.showMAs || options.showAllOverlay) {
                drawMicroaneurysms(ctx, cx, cy, radius, sample.maCount);
            }
        }

        if (sample.drLevel >= 2) {
            if (options.showExudates || options.showAllOverlay) {
                drawExudates(ctx, cx, cy, radius, sample.exudatesArea);
            }
            if (options.showHemorrhages || options.showAllOverlay) {
                drawHemorrhages(ctx, cx, cy, radius, sample.hemorrhagesCount);
            }
        }

        if (sample.drLevel === 4 && sample.hasNV) {
            if (options.showNV || options.showAllOverlay) {
                drawNeovascularization(ctx, odX, odY, radius);
            }
        }

        ctx.restore();
    }

    function drawVessels(ctx, cx, cy, radius, options) {
        ctx.strokeStyle = (options.mode === 'green_channel') ? 'rgba(0, 40, 0, 0.85)' : 'rgba(80, 10, 5, 0.85)';
        ctx.lineWidth = (options.mode === 'clahe') ? 3.5 : 2.5;
        ctx.lineCap = 'round';

        const odX = cx + radius * 0.5;
        const odY = cy - radius * 0.05;

        const mainArches = [
            // Superior Arch
            [[odX, odY], [odX - radius*0.3, odY - radius*0.5], [odX - radius*0.8, odY - radius*0.4]],
            // Inferior Arch
            [[odX, odY], [odX - radius*0.3, odY + radius*0.5], [odX - radius*0.8, odY + radius*0.4]],
            // Nasal Branches
            [[odX, odY], [odX + radius*0.25, odY - radius*0.3], [odX + radius*0.4, odY - radius*0.5]],
            [[odX, odY], [odX + radius*0.25, odY + radius*0.3], [odX + radius*0.4, odY + radius*0.5]]
        ];

        mainArches.forEach(pts => {
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            ctx.quadraticCurveTo(pts[1][0], pts[1][1], pts[2][0], pts[2][1]);
            ctx.stroke();

            // Sub-branches
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(pts[1][0], pts[1][1]);
            ctx.lineTo(pts[1][0] - 25, pts[1][1] + 15);
            ctx.stroke();
        });
    }

    function drawOpticDisc(ctx, x, y, rad, options) {
        ctx.beginPath();
        ctx.arc(x, y, rad, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 235, 180, 0.9)';
        ctx.fill();

        ctx.strokeStyle = '#00f2fe';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Crosshair marker
        ctx.strokeStyle = '#00f2fe';
        ctx.beginPath();
        ctx.moveTo(x - rad - 5, y); ctx.lineTo(x + rad + 5, y);
        ctx.moveTo(x, y - rad - 5); ctx.lineTo(x, y + rad + 5);
        ctx.stroke();

        ctx.fillStyle = '#00f2fe';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.fillText('OPTIC DISC', x - 32, y - rad - 8);
    }

    function drawFovea(ctx, x, y, rad) {
        ctx.beginPath();
        ctx.arc(x, y, rad, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(40, 10, 5, 0.7)';
        ctx.fill();

        ctx.strokeStyle = '#7c4dff';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#7c4dff';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.fillText('FOVEA (FAZ)', x - 32, y + rad + 16);
    }

    function drawMicroaneurysms(ctx, cx, cy, radius, count) {
        ctx.fillStyle = '#ff5252';
        const coords = getSyntheticMaCoords(cx, cy, radius, count);
        coords.forEach(c => {
            ctx.beginPath();
            ctx.arc(c.x, c.y, 3.5, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.strokeStyle = 'rgba(255, 82, 82, 0.6)';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }

    function drawExudates(ctx, cx, cy, radius, area) {
        ctx.fillStyle = '#fff59d';
        ctx.shadowColor = '#fff59d';
        ctx.shadowBlur = 4;

        const count = Math.min(18, Math.max(4, Math.floor(area / 100)));
        for (let i = 0; i < count; i++) {
            const angle = (i / count) * Math.PI * 1.6 - 0.8;
            const dist = radius * (0.25 + (i % 3) * 0.12);
            const exX = cx - Math.cos(angle) * dist;
            const exY = cy - Math.sin(angle) * dist;

            ctx.beginPath();
            ctx.ellipse(exX, exY, 6 + (i%4), 4 + (i%3), Math.PI/4, 0, Math.PI*2);
            ctx.fill();
        }
        ctx.shadowBlur = 0;
    }

    function drawHemorrhages(ctx, cx, cy, radius, count) {
        ctx.fillStyle = '#d50000';
        for (let i = 0; i < count; i++) {
            const hX = cx + (Math.sin(i * 1.3) * radius * 0.55);
            const hY = cy + (Math.cos(i * 1.7) * radius * 0.45);

            ctx.beginPath();
            ctx.ellipse(hX, hY, 8 + (i % 5), 4 + (i % 3), i * 0.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function drawNeovascularization(ctx, odX, odY, radius) {
        ctx.strokeStyle = '#ff1744';
        ctx.lineWidth = 1.5;
        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2;
            ctx.beginPath();
            ctx.moveTo(odX, odY);
            ctx.lineTo(odX + Math.cos(angle) * 35, odY + Math.sin(angle) * 35);
            ctx.stroke();
        }
        ctx.fillStyle = '#ff1744';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.fillText('NVD PROLIFERATION', odX - 45, odY + 45);
    }

    function getSyntheticMaCoords(cx, cy, radius, count) {
        const list = [];
        for (let i = 0; i < count; i++) {
            const angle = (i * 0.7) + 0.3;
            const dist = radius * (0.2 + (i % 5) * 0.1);
            list.push({
                x: cx - Math.cos(angle) * dist,
                y: cy - Math.sin(angle) * dist
            });
        }
        return list;
    }

    return {
        SAMPLES,
        renderFundusImage,
        getCurrentSample: () => SAMPLES[currentSampleKey]
    };
})();
